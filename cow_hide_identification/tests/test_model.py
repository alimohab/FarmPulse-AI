"""Tests for the network, loss, mining, metrics, checkpointing and a short training run."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from cowid.reid.backbone import build_backbone
from cowid.reid.losses import SemiHardTripletLoss
from cowid.reid.siamese_network import SiameseNetwork
from cowid.reid.triplet_mining import mine_semi_hard_triplets
from cowid.training.metrics import retrieval_metrics
from cowid.utils.checkpoint_utils import (BEST_CHECKPOINT_NAME, LAST_CHECKPOINT_NAME, BestMetricTracker,
                                          load_checkpoint, load_model_for_inference, save_checkpoint)


@pytest.fixture(scope="module")
def resnet50_model():
    torch.manual_seed(0)
    return SiameseNetwork("resnet50", embedding_dim=128, pretrained=False).eval()


def test_resnet50_embeddings_have_configured_dim_and_unit_norm(resnet50_model):
    with torch.no_grad():
        emb = resnet50_model(torch.randn(3, 3, 96, 96))
    assert emb.shape == (3, 128)
    assert torch.allclose(emb.norm(dim=1), torch.ones(3), atol=1e-5)


def test_pair_forward_shares_weights(resnet50_model):
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        e1, e2 = resnet50_model(x, x.clone())
        sim = resnet50_model.similarity(x, x.clone())
    assert torch.allclose(e1, e2, atol=1e-6)
    assert torch.allclose(sim, torch.ones(2), atol=1e-5)


def test_embedding_dim_and_backbone_are_configurable():
    model = SiameseNetwork("resnet18", embedding_dim=64, hidden_dim=0, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 64)
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone("not_a_backbone", pretrained=False)


def test_frozen_backbone_receives_no_gradients():
    model = SiameseNetwork("resnet18", embedding_dim=16, hidden_dim=32, pretrained=False)
    model.set_backbone_trainable(False)
    model.train()
    assert not model.backbone.training and model.head.training
    model(torch.randn(4, 3, 32, 32)).sum().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is not None for p in model.head.parameters())


def test_semi_hard_miner_respects_identity_and_view():
    emb = F.normalize(torch.randn(8, 16), dim=1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    views = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    t = mine_semi_hard_triplets(emb, labels, views, cross_view=False)
    assert len(t) == 8  # each image has exactly one same-cow, same-view partner
    assert (labels[t.anchor] == labels[t.positive]).all() and (views[t.anchor] == views[t.positive]).all()
    assert (labels[t.anchor] != labels[t.negative]).all() and (t.anchor != t.positive).all()
    assert len(mine_semi_hard_triplets(emb, labels, views, cross_view=True)) == 8 * 3


def test_semi_hard_miner_picks_closest_negative_beyond_positive():
    def unit(deg):
        return [math.cos(math.radians(deg)), math.sin(math.radians(deg))]
    # anchor 0deg, positive 30deg; negatives at 10deg (hard), 45deg (semi-hard), 90deg (easy)
    emb = torch.tensor([unit(0), unit(30), unit(10), unit(45), unit(90)])
    labels = torch.tensor([0, 0, 1, 2, 3])
    t = mine_semi_hard_triplets(emb, labels)
    first = (t.anchor == 0).nonzero()[0].item()
    assert t.negative[first].item() == 3


def test_triplet_loss_values():
    loss_fn = SemiHardTripletLoss(margin=0.2)
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    loss, stats = loss_fn(separated, labels)
    assert stats["num_triplets"] == 4 and loss.item() == pytest.approx(0.0, abs=1e-6)
    collapsed = torch.tensor([[1.0, 0.0]] * 4, requires_grad=True)
    loss, _ = loss_fn(collapsed, labels)
    assert loss.item() == pytest.approx(0.2, abs=1e-5)
    loss.backward()


def test_retrieval_metrics_perfect_and_same_view():
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    g_labels, g_views = torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1])
    queries = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    m = retrieval_metrics(queries, torch.tensor([0, 1]), gallery, g_labels, ks=(1, 2, 5),
                          query_views=torch.tensor([0, 0]), gallery_views=g_views, same_view=True)
    assert m["top1"] == 1.0 and m["recall@2"] == 1.0 and m["mAP"] == 1.0
    assert "recall@5" not in m  # only 2 same-view gallery candidates


def test_best_tracker_only_accepts_improvements():
    tracker = BestMetricTracker("max", min_delta=0.01)
    assert tracker.update(0.50, 1) and tracker.update(0.60, 2)
    assert not tracker.update(0.605, 3) and not tracker.update(0.40, 4) and not tracker.update(None, 5)
    assert tracker.best == 0.60 and tracker.best_epoch == 2 and tracker.num_bad_epochs == 3
    loss_tracker = BestMetricTracker("min")
    assert loss_tracker.update(1.0, 1) and loss_tracker.update(0.5, 2) and not loss_tracker.update(0.7, 3)


def test_checkpoint_roundtrip_for_inference_and_training(tmp_path):
    model = SiameseNetwork("resnet18", embedding_dim=32, hidden_dim=64, pretrained=False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    model(torch.randn(4, 3, 32, 32)).sum().backward()
    optimizer.step()
    model.eval()
    path = save_checkpoint(tmp_path / "ckpt.pth", model, epoch=3, optimizer=optimizer, best_metric=0.75,
                           monitor="top1", config={"data": {"image_size": [32, 32]}})
    ckpt = load_checkpoint(path)
    for key in ("model_state_dict", "optimizer_state_dict", "epoch", "best_metric", "model_config", "embedding_dim"):
        assert key in ckpt
    assert ckpt["epoch"] == 3 and ckpt["best_metric"] == 0.75 and ckpt["embedding_dim"] == 32
    loaded, _ = load_model_for_inference(path)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x), atol=1e-6)
    restored = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
    restored.load_state_dict(ckpt["optimizer_state_dict"])
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pth")


def test_training_writes_best_and_last_checkpoints_and_resumes(make_dataset, small_config):
    from cowid.training.train import train

    cfg = small_config(make_dataset(cows=3, views=("left", "right"), images_per_view=6))
    summary = train(cfg)
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    last = load_checkpoint(f"{ckpt_dir}/{LAST_CHECKPOINT_NAME}")
    best = load_checkpoint(f"{ckpt_dir}/{BEST_CHECKPOINT_NAME}")
    assert last["epoch"] == 2 and len(last["history"]) == 2
    top1 = [row["top1"] for row in last["history"]]
    assert best["best_metric"] == max(top1) and best["epoch"] == summary["best_epoch"]
    assert best["epoch"] == 1 + top1.index(max(top1))  # first epoch reaching the best value

    cfg["training"]["epochs"] = 3
    train(cfg, resume=True)
    resumed = load_checkpoint(f"{ckpt_dir}/{LAST_CHECKPOINT_NAME}")
    assert resumed["epoch"] == 3 and len(resumed["history"]) == 3
    best_after = load_checkpoint(f"{ckpt_dir}/{BEST_CHECKPOINT_NAME}")
    assert best_after["best_metric"] == max(row["top1"] for row in resumed["history"])

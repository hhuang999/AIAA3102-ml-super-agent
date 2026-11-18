"""Epoch-boundary equivalence with shuffled samples, stochastic worker transforms and GPU RNG."""

import random
import numpy as np
import pytest


class AugmentedRows:
    def __len__(self):
        return 12

    def __getitem__(self, i):
        import torch

        return torch.tensor(
            [i / 12, random.random(), np.random.rand(), torch.rand(()).item()]
        ), i % 2


@pytest.mark.slow
def test_two_workers_and_cuda_resume_at_epoch_boundary(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed to verify accelerator RNG")
    from superagent.training import save_torch, restore_torch

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SUPERAGENT_CHECKPOINT_DIR", str(tmp_path / "checkpoint"))
    monkeypatch.setenv("SUPERAGENT_TRIAL_SIGNATURE", "augmented-gpu")
    monkeypatch.setenv("SUPERAGENT_RUNG", "0")
    monkeypatch.delenv("SUPERAGENT_RESUME_DIR", raising=False)
    torch.manual_seed(19)

    def init():
        model = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(4, 2)).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        return model, opt, torch.Generator().manual_seed(42)

    def epoch(model, opt, g):
        seen = []
        for x, y in torch.utils.data.DataLoader(
            AugmentedRows(),
            shuffle=True,
            generator=g,
            batch_size=4,
            num_workers=2,
            persistent_workers=False,
        ):
            seen.append(x)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x.cuda()), y.cuda())
            loss.backward()
            opt.step()
        return torch.cat(seen)

    m, o, g = init()
    restore_torch(m, o, generator=g)
    epoch(m, o, g)
    save_torch(m, o, epoch=1, next_seconds=2, delivery_seconds=1, generator=g)
    expected = epoch(m, o, g)
    m2, o2, g2 = init()
    monkeypatch.setenv("SUPERAGENT_RESUME_DIR", str(tmp_path / "checkpoint"))
    assert restore_torch(m2, o2, generator=g2)[0] == 1
    actual = epoch(m2, o2, g2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(m.parameters(), m2.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

import random
import numpy as np
import pytest
from superagent.training import save_torch,restore_torch


def test_restored_optimizer_schedule_and_randomness_match(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    for k,v in dict(CHECKPOINT_DIR=str(tmp_path/"state"),TRIAL_SIGNATURE="sig",RUNG="0").items():
        monkeypatch.setenv("SUPERAGENT_"+k,v)
    def init():
        model=torch.nn.Linear(4,2)
        opt=torch.optim.AdamW(model.parameters())
        return model,opt,torch.optim.lr_scheduler.StepLR(opt,1),torch.Generator().manual_seed(17)
    def step(model,opt,sched,gen):
        x=torch.rand(4,4,generator=gen)+np.random.rand()+random.random()
        opt.zero_grad()
        model(x).sum().backward();opt.step();sched.step()
    m,o,s,g=init();step(m,o,s,g)
    save_torch(m,o,epoch=1,next_seconds=3,delivery_seconds=2,scheduler=s,generator=g,sampler_state={"batch":0})
    step(m,o,s,g)
    n,p,t,h=init()
    monkeypatch.setenv("SUPERAGENT_RESUME_DIR",str(tmp_path/"state"))
    assert restore_torch(n,p,scheduler=t,generator=h)==(1,{"batch":0})
    step(n,p,t,h)
    assert torch.equal(n.weight,m.weight) and torch.equal(n.bias,m.bias)

from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn
from rttp_grid.config import GridConfig
from rttp_grid.models import FunctionalModel, flatten_params
from rttp_grid.runner import (AdamState, AdamTangent, adam_step, adam_tangent_step,
                              sgd_step, sgd_tangent_step, make_context, summarize_outputs)
from rttp_grid.utils import rel_error

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class ProductionTangents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        model = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2)).double()
        self.model = FunctionalModel(model)
        self.theta = flatten_params(model).detach()
        self.batches = [(torch.randn(4, 4, dtype=torch.float64), torch.tensor([0, 1, 0, 1])) for _ in range(3)]
        self.q = torch.randn_like(self.theta) * 0.01

    def test_sgd_production_three_step_tangent(self):
        th = self.theta.clone()
        
        torch.manual_seed(44); m0 = torch.randn_like(th) * 0.01
        def trajectory(alpha):
            t, mem = self.theta.clone(), m0 + alpha*self.q
            for x,y in self.batches:t,mem=sgd_step(self.model,t,mem,x,y,0.03,0.9,0.001)
            return t
        t,mem,dt,dm=self.theta.clone(),m0.clone(),torch.zeros_like(th),self.q.clone()
        for x,y in self.batches:t,mem,dt,dm=sgd_tangent_step(self.model,t,mem,dt,dm,x,y,0.03,0.9,0.001)
        fd=(trajectory(1e-4)-trajectory(-1e-4))/(2e-4)
        self.assertLess(rel_error(dt,fd),1e-6)

    def test_adam_production_three_step_memory_channels(self):
        ctx=SimpleNamespace(cfg=GridConfig(),device=torch.device('cpu'),dtype=torch.float64)
        base=AdamState(self.theta.clone(),torch.randn_like(self.theta)*0.01,torch.rand_like(self.theta)*0.01+0.001,17)
        for component in ['m','rms_global']:
            with self.subTest(component=component):
                q=self.q if component=='m' else torch.ones_like(self.theta)
                tangent=AdamTangent(torch.zeros_like(q),q.clone() if component=='m' else torch.zeros_like(q),torch.zeros_like(q) if component=='m' else 2*base.v*q)
                nominal=base
                for x,y in self.batches:nominal,tangent=adam_tangent_step(ctx,self.model,nominal,tangent,x,y,3e-4)
                def trajectory(alpha):
                    st=AdamState(base.theta.clone(),base.m+alpha*q if component=='m' else base.m.clone(),base.v.clone() if component=='m' else base.v*torch.exp(2*alpha*q),base.step)
                    for x,y in self.batches:st=adam_step(ctx,self.model,st,x,y,3e-4)
                    return st.theta
                fd=(trajectory(1e-4)-trajectory(-1e-4))/(2e-4)
                self.assertLess(rel_error(tangent.dtheta,fd),1e-5)


class Packaging(unittest.TestCase):
    def test_all_configs_construct(self):
        from rttp_grid.amortization import AmortizationConfig
        from rttp_grid.surrogate_baselines import SurrogateBaselineConfig
        from rttp_grid.candidate_selection import CandidateSelectionConfig
        classes={'full_grid':GridConfig,'smoke':GridConfig,'amortization':AmortizationConfig,'surrogate_baselines':SurrogateBaselineConfig,'candidate_selection':CandidateSelectionConfig}
        for name,cls in classes.items():cls(**json.loads((ROOT/f'configs/{name}.json').read_text()))

    def test_cached_scientific_setting_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=GridConfig(root_name=tmp,use_google_drive=False)
            make_context(cfg)
            with self.assertRaisesRegex(ValueError,'different scientific settings'):
                make_context(replace(cfg,batch_size=7))

    def test_summary_uses_requested_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=GridConfig(root_name='nonexistent_old_relative_root',use_google_drive=False,run_figures=False)
            cfg.to_json(Path(tmp)/'config.json')
            summarize_outputs(Path(tmp))
            self.assertTrue((Path(tmp)/'tables').is_dir())
            self.assertFalse((Path.cwd()/'nonexistent_old_relative_root').exists())

    def test_archived_table1_values_match_manuscript(self):
        table=ROOT/'reference_results/full_grid/reanalysis/tables'
        sgd=pd.read_csv(table/'sgd_h40_main_table.csv')
        expected={('split_cifar10','smallcnn'):0.0375,('split_cifar10','resnet18'):0.0393,('split_tinyimagenet','smallcnn'):0.0246,('split_tinyimagenet','resnet18'):0.0217}
        for (dataset,arch),value in expected.items():
            row=sgd[(sgd.dataset==dataset)&(sgd.architecture==arch)].iloc[0]
            self.assertEqual(round(float(row.endpoint_error),4),value)
        adam=pd.read_csv(table/'adam_h20_component_table.csv')
        expected={('split_cifar10','smallcnn'):0.0114,('split_cifar10','resnet18'):0.0481,('split_tinyimagenet','smallcnn'):0.0001,('split_tinyimagenet','resnet18'):0.2241}
        for (dataset,arch),value in expected.items():
            row=adam[(adam.dataset==dataset)&(adam.architecture==arch)&(adam.memory_component=='m')].iloc[0]
            self.assertEqual(round(float(row.endpoint_error),4),value)

    def test_anonymous_notebook_metadata_and_compilation(self):
        for p in ROOT.rglob('*.ipynb'):
            nb=json.loads(p.read_text())
            for c in nb['cells']:
                if c['cell_type']=='code':
                    self.assertFalse(c['outputs']);self.assertIsNone(c['execution_count'])
        for p in ROOT.rglob('*.py'):compile(p.read_text(),str(p),'exec')


class OfflinePipelines(unittest.TestCase):
    pass
    def task_data(self,*args):
        from rttp_grid.data import TaskData
        from torch.utils.data import DataLoader, TensorDataset
        g=torch.Generator().manual_seed(22)
        ds=TensorDataset(torch.randn(8,4,generator=g),torch.arange(8)%2)
        loader=DataLoader(ds,batch_size=4,shuffle=False)
        return TaskData(loader,loader,loader,loader,2,4)

    def tiny_model(self,*args):
        return nn.Sequential(nn.Linear(4,3),nn.Tanh(),nn.Linear(3,2))

    def test_grid_training_response_summary_and_resume(self):
        from unittest.mock import patch
        from rttp_grid.runner import run_grid
        with tempfile.TemporaryDirectory() as tmp, patch('rttp_grid.runner.make_task_data',self.task_data), patch('rttp_grid.runner.build_model',self.tiny_model):
            cfg=GridConfig(**json.loads((ROOT/'configs/smoke.json').read_text()))
            cfg=replace(cfg,root_name=tmp,run_preflight_audit=False,run_dataset_probe=False,dtype='float64')
            run_grid(cfg)
            raw=pd.read_csv(Path(tmp)/'tables/response_raw.csv')
            self.assertTrue(np.isfinite(raw[raw.response_model=='time_varying'].endpoint_error).all())
            quality=pd.read_csv(Path(tmp)/'tables/quality_report.csv')
            self.assertTrue(quality[quality.blocking].passed.all())
            run_grid(cfg)
            resumed=pd.read_csv(Path(tmp)/'tables/response_raw.csv')
            pd.testing.assert_frame_equal(raw,resumed)

    def test_auxiliary_experiment_pipelines(self):
        from unittest.mock import patch
        from contextlib import ExitStack
        from rttp_grid.amortization import AmortizationConfig,run_amortization_experiment
        from rttp_grid.surrogate_baselines import SurrogateBaselineConfig,run_surrogate_baseline_experiment
        from rttp_grid.candidate_selection import CandidateSelectionConfig,run_candidate_selection_experiment
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch('rttp_grid.runner.build_model',self.tiny_model))
            for name in ['amortization','surrogate_baselines','candidate_selection']:
                stack.enter_context(patch(f'rttp_grid.{name}.make_task_data',self.task_data))
            common=dict(horizon=3,response_rank=2,num_workers=0,batch_size=4,boundary_epochs_smallcnn_cifar=1,use_google_drive=False,dtype='float64')
            jobs=[
                (AmortizationConfig(root_name=str(Path(tmp)/'amort'),candidate_counts=(2,3),repeats=1,warmup=0,**common),run_amortization_experiment),
                (SurrogateBaselineConfig(root_name=str(Path(tmp)/'surrogates'),candidate_count=3,periodic_refresh_periods=(2,),low_rank_response_ranks=(1,),truncated_tail_lengths=(1,),**common),run_surrogate_baseline_experiment),
                (CandidateSelectionConfig(root_name=str(Path(tmp)/'selection'),candidate_count=3,topk_values=(1,2),**common),run_candidate_selection_experiment),
            ]
            for cfg,run in jobs:
                with self.subTest(experiment=type(cfg).__name__):
                    out=run(cfg)
                    summaries=list((out/'tables').glob('*summary*.csv'))
                    self.assertTrue(summaries)
                    self.assertTrue(all(p.stat().st_size>0 for p in summaries))

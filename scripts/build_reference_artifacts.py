
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/reference_artifacts'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ref=Path(__file__).resolve().parents[1]/'reference_results'
    tables=ref/'full_grid/reanalysis/tables'
    sgd=pd.read_csv(tables/'sgd_h40_main_table.csv')
    adam=pd.read_csv(tables/'adam_h20_component_table.csv');adam=adam[adam.memory_component.eq('m')]
    tab=pd.concat([sgd,adam],ignore_index=True)[['dataset','optimizer','architecture','horizon','endpoint_error','endpoint_cosine']]
    tab.to_csv(args.output/'table1_and_table3.csv',index=False,float_format='%.4f')
    summary=pd.read_csv(ref/'full_grid/tables/response_summary_by_scale.csv')
    seed=pd.read_csv(ref/'full_grid/tables/seed_level_response.csv')
    ops=pd.read_csv(tables/'frozen_ci_operating_points.csv')
    def save(fig,name):
        fig.tight_layout();fig.savefig(args.output/(name+'.pdf'));fig.savefig(args.output/(name+'.png'),dpi=180);plt.close(fig)
    
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for row,opt,comp in [(0,'heavy_ball','momentum'),(1,'adam','m')]:
        for _,op in ops[(ops.optimizer==opt)&(ops.memory_component==comp)].iterrows():
            panel=seed[(seed.optimizer==opt)&(seed.memory_component==comp)&(seed.dataset==op.dataset)&(seed.architecture==op.architecture)&(seed.response_model=='time_varying')&np.isclose(seed.learning_rate,op.learning_rate)&np.isclose(seed.radius_scale,op.radius_scale)]
            if opt=='heavy_ball':panel=panel[np.isclose(panel.momentum,op.momentum)]
            for col,metric in enumerate(['endpoint_error','endpoint_cosine']):
                g=panel.groupby('horizon')[metric].agg(['mean','std','count']);ci=2.776*g['std']/np.sqrt(g['count'])
                line,=axs[row,col].plot(g.index,g['mean'],marker='o',label=f'{op.dataset} / {op.architecture}')
                axs[row,col].fill_between(g.index,g['mean']-ci,g['mean']+ci,alpha=.15,color=line.get_color())
                axs[row,col].set(xlabel='Horizon',ylabel=metric,title=opt)
        axs[row,0].legend(fontsize=6)
    save(fig,'figure1_response')
    
    panel=summary[(summary.dataset=='split_cifar10')&(summary.architecture=='resnet18')&(summary.optimizer=='adam')&(summary.memory_component=='rms_global')&np.isclose(summary.learning_rate,3e-4)]
    tv=panel[panel.response_model=='time_varying']
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axs,['endpoint_error','endpoint_cosine']):
        pivot=tv.pivot(index='radius_scale',columns='horizon',values=metric).sort_index()
        im=ax.imshow(pivot.to_numpy(),aspect='auto');ax.set_xticks(range(len(pivot.columns)),pivot.columns);ax.set_yticks(range(len(pivot.index)),[f'{v:g}' for v in pivot.index]);ax.set(xlabel='Horizon',ylabel='RMS perturbation',title=metric);fig.colorbar(im,ax=ax)
    save(fig,'figure2_predictive_regime')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for model in ['time_varying','frozen_boundary']:
        p=panel[(panel.response_model==model)&np.isclose(panel.radius_scale,.1)].sort_values('horizon')
        for ax,metric in zip(axs,['endpoint_error','endpoint_cosine']):ax.plot(p.horizon,p[metric],marker='o',label=model);ax.set(xlabel='Horizon',ylabel=metric);ax.legend()
    axs[0].set_yscale('log');save(fig,'figure3_frozen_vs_timevarying')
    
    rows=[]
    for dataset,lr in [('split_cifar10',3e-4),('split_tinyimagenet',1e-4)]:
        for arch in ['smallcnn','resnet18']:
            for comp,scale in [('m',.01),('rms_global',.1)]:
                p=summary[(summary.dataset==dataset)&(summary.architecture==arch)&(summary.optimizer=='adam')&(summary.response_model=='time_varying')&(summary.horizon==10)&(summary.memory_component==comp)&np.isclose(summary.learning_rate,lr)&np.isclose(summary.radius_scale,scale)]
                r=p.iloc[0];rows.append({'setting':dataset+' / '+arch,'component':comp,'endpoint_error':r.endpoint_error,'endpoint_cosine':r.endpoint_cosine})
    data=pd.DataFrame(rows);data.to_csv(args.output/'figure5_values.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axs,['endpoint_error','endpoint_cosine']):
        p=data.pivot(index='setting',columns='component',values=metric);im=ax.imshow(p.to_numpy(),aspect='auto');ax.set_xticks(range(len(p.columns)),p.columns);ax.set_yticks(range(len(p.index)),p.index,fontsize=7);ax.set_title(metric);fig.colorbar(im,ax=ax)
    save(fig,'figure5_generalization')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axs,['optimizer_response_cifar10_rttp','optimizer_response_cifar10_rttp_sgd_resnet18_paper']):
        p=pd.read_csv(ref/'legacy'/name/'table_policy_summary.csv')
        arch='smallcnn' if name=='optimizer_response_cifar10_rttp' else 'resnet18'
        p=p[p.architecture==arch]
        ax.scatter(p.B_acc,p.A_acc)
        for _,r in p.iterrows():ax.annotate(r.policy,(r.B_acc,r.A_acc),fontsize=7)
        ax.set(xlabel='Task-B accuracy',ylabel='Task-A accuracy',title=arch)
    save(fig,'figure6_policy_tradeoff')
    print('Reference artifacts written to',args.output)


if __name__=='__main__':main()

# coding: utf-8
from pathlib import Path
import json, argparse, datetime
parser=argparse.ArgumentParser();parser.add_argument('--draft',action='store_true');args=parser.parse_args()
base=Path(__file__).resolve().parents[3]
run=base/'报告/随机场景_20261026'
out=base/'提交/cucu_初赛提交材料'
template=json.loads((run/'官方模板/灵波2026/初赛评测结果模板_JSON版.json').read_text())
clean=json.loads((base/'报告/FP32全任务/结果.json').read_text())
randomized=json.loads((run/'snapshot.json').read_text())
backup=json.loads((run/'采集状态.json').read_text())
assert clean['complete'] and clean['completed_tasks']==50 and clean['completed_task_rollouts']==500
assert clean['completed_task_successes']==263
assert set(template)=={'schema_version','team_id','results'} and template['schema_version']==1
template['team_id']='1382038'
counts={}
for setting,result in [('clean',clean),('randomized',randomized)]:
    tasks={t['task']:t for t in result['tasks']}
    assert set(tasks)==set(template['results'][setting]),setting
    for name,row in tasks.items():
        episodes=row['episodes']
        attempts=len(episodes);successes=sum(bool(e['success']) for e in episodes)
        assert attempts==row['policy_rollouts'] and successes==row['successes'],name
        assert attempts<=10 and 0<=successes<=attempts
        assert len({e['seed'] for e in episodes})==attempts,name
        if not args.draft:assert row['completed'] and attempts==10,name
        template['results'][setting][name]={'attempts':attempts,'successes':successes}
    counts[setting]={'tasks':50,'attempts':sum(v['attempts'] for v in template['results'][setting].values()),'successes':sum(v['successes'] for v in template['results'][setting].values())}
if not args.draft:
    assert randomized['complete'] and backup['backup_complete'] and backup['videos_verified']==500
    assert randomized['exit_code']==0 and not randomized['issues']
name='results_draft.json' if args.draft else 'results.json'
path=out/'01_评测结果'/name
path.write_text(json.dumps(template,ensure_ascii=False,indent=2)+'\n')
if not args.draft:
    (path.parent/'results_draft.json').unlink(missing_ok=True)
receipt={'created_hong_kong':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'team_id':'1382038','schema_matches_downloaded_official_template':True,'draft':args.draft,'counts':counts,'path':str(path),'all_trials_are_actual_policy_outcomes':True}
(run/('官方结果格式草稿校验.json' if args.draft else '官方结果格式校验.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False))

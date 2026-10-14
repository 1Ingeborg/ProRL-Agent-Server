"""Synchronous remote Polar rollout for Slime's train.py.

Polling avoids callbacks to arbitrary AutoDL ports. Prompt groups run with
bounded concurrency; all finish before training and weight synchronization.
"""
import asyncio, hashlib, json, logging, os
from pathlib import Path
import httpx
from slime_bridge import rollout as upstream
from slime_bridge.config import resolve_polar_slime_config
ROOT=Path(os.environ.get("POLAR_HOME",str(Path(__file__).resolve().parents[1])))
logger=logging.getLogger(__name__)
def generate(args,rollout_id,data_source,evaluation=False):
    if evaluation:
        return upstream.generate_rollout_polar_async(args,rollout_id,data_source,evaluation=True)
    groups=data_source.get_samples(int(args.rollout_batch_size))
    if len(groups)!=int(args.rollout_batch_size):raise RuntimeError("Insufficient prompt groups")
    cfg=resolve_polar_slime_config(args)
    run_dir=ROOT/"runs/training"/str(args.polar_run_id)
    cache_dir=run_dir/"task-results";cache_dir.mkdir(parents=True,exist_ok=True)
    smoke_hash=hashlib.sha256(Path(args.prompt_data).read_bytes()).hexdigest()
    reused=[]
    async def run():
        async with httpx.AsyncClient(timeout=cfg.request_timeout,trust_env=False) as client:
            limit=max(1,int(cfg.max_async_level))
            semaphore=asyncio.Semaphore(limit)
            async def one(position,group):
              async with semaphore:
                payload=upstream._build_task_payload(args=args,config=cfg,group=group,rollout_id=rollout_id,task_position=position)
                # Dataset export is frozen; per-run IDs belong to the launch.
                payload.setdefault("metadata",{})["run_id"]=str(args.polar_run_id)
                payload["agent"].setdefault("settings",{})["run_id"]=str(args.polar_run_id)
                upstream._attach_scheduler_metadata(payload,group_id=position,policy_version=rollout_id,rollout_step=rollout_id)
                cache_path=cache_dir/(payload["task_id"]+".json")
                payload_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                if cache_path.exists():
                    cached=json.loads(cache_path.read_text())
                    if cached["smoke_sha256"]!=smoke_hash or cached["policy_version"]!=rollout_id:
                        raise RuntimeError("Cached rollout data/policy identity changed")
                    if cached.get("payload_sha256") not in {None,payload_hash}:
                        raise RuntimeError("Cached rollout request changed")
                    result=upstream.TaskResult.model_validate(cached["result"])
                    if result.task_id!=payload["task_id"]:
                        raise RuntimeError("Cached task identity changed")
                    for session in result.results:
                        if session.metadata.get("run_id")!=str(args.polar_run_id):
                            raise RuntimeError("Cached session run identity changed")
                    reused.append(result.task_id)
                    logger.info("Reusing %s completed real sessions for %s",len(result.results),result.task_id)
                else:
                    result=await asyncio.wait_for(
                        upstream._submit_and_wait_for_task(client,cfg.rollout_server_url,payload),
                        timeout=cfg.request_timeout)
                if result.status!="completed" or len(result.results)!=len(group):
                    raise RuntimeError("Incomplete Polar task: "+result.task_id)
                if any(str(getattr(x.status,"value",x.status))!="COMPLETED" for x in result.results):
                    raise RuntimeError("Polar session failed: "+result.task_id)
                samples=upstream._convert_task_result_to_samples(cfg,result,group,max_tokens=upstream._resolve_max_tokens(args))
                if not samples or not upstream._has_trainable_tokens(samples):
                    raise RuntimeError("No trainable token traces: "+result.task_id)
                reason=upstream._low_complete_accept_fraction_rejection_reason(cfg,result,samples)
                if reason:raise RuntimeError(str(reason))
                if not cache_path.exists():
                    cache_path.write_text(json.dumps({"smoke_sha256":smoke_hash,"policy_version":rollout_id,
                        "payload_sha256":payload_hash,"result":result.model_dump(mode="json")})+"\n")
                upstream._annotate_accepted_samples(samples,accepted_rollout_id=rollout_id,policy_version=rollout_id,staleness=0,scheduler_group_id=position)
                return result,samples
            jobs=[asyncio.create_task(one(position,group)) for position,group in enumerate(groups)]
            try:
                pairs=await asyncio.gather(*jobs)
            except BaseException:
                for job in jobs:job.cancel()
                await asyncio.gather(*jobs,return_exceptions=True)
                raise
            return [p[0] for p in pairs],[p[1] for p in pairs]
    results,data=asyncio.run(run())
    flat=[s for g in data for s in g]
    rewards=[upstream._extract_sample_reward(s,cfg.reward_key) for s in flat]
    metrics=upstream._polar_extra_metrics(flat,rewards,cfg.reward_key)
    metrics["polar/reused_session_count"]=float(sum(len(r.results) for r in results if r.task_id in reused))
    reports=[]
    for sample in flat:
        polar_meta=(getattr(sample,"metadata",None) or {}).get("polar",{})
        report=((polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}).get("report") or {}
        if isinstance(report,dict): reports.append(report)
    if reports:
        metrics["agent/empty_patch_ratio"]=sum(bool(r.get("empty_generation")) for r in reports)/len(reports)
        metrics["agent/session_count"]=float(len(reports))
    if "TENSORBOARD_DIR" in os.environ:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(os.environ["TENSORBOARD_DIR"])
        for key,value in metrics.items(): writer.add_scalar("rollout/"+key,float(value),rollout_id)
        writer.flush(); writer.close()
    path=ROOT/"runs/training"/str(args.polar_run_id);path.mkdir(parents=True,exist_ok=True)
    (path/("rollout-"+str(rollout_id)+".json")).write_text(json.dumps({
        "run_id":args.polar_run_id,"rollout_id":rollout_id,"task_ids":[x.task_id for x in results],
        "reused_task_ids":reused,
        "trace_count":len(flat),"session_count":sum(len(x.results) for x in results),
        "rewards":rewards,"empty_patch_count":sum(bool(r.get("empty_generation")) for r in reports),
        "policy_version":rollout_id,"model_calls":True},indent=2)+"\n")
    return upstream._load_rollout_train_output_type()(samples=data,metrics=metrics)

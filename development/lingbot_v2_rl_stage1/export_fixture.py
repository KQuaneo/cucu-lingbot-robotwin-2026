"""Export two aligned raw observations from the existing LeRobot v3 episode 0."""
import argparse
import json
from pathlib import Path


def main():
    import av
    import numpy as np
    import pyarrow.parquet as pq
    from stage1 import raw_observations

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.dataset
    info = json.loads((root / "meta/info.json").read_text())
    # This fixture exporter deliberately supports only the verified first episode.
    ep = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").slice(0, 1).to_pylist()[0]
    if ep["episode_index"] != 0 or ep["dataset_from_index"] != 0:
        raise ValueError("Expected episode 0 at dataset row 0")
    path = root / info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
    rows = pq.read_table(path).slice(0, 31).to_pylist()
    selected = [rows[0], rows[30]]
    if any(r["episode_index"] != 0 for r in selected):
        raise ValueError("Episode too short")
    tasks = {r["task_index"]: r["task"] for r in pq.read_table(root / "meta/tasks.parquet").to_pylist()}
    images, timestamps = {}, {}
    for key in ("observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"):
        prefix = "videos/" + key
        path = root / info["video_path"].format(video_key=key,
                chunk_index=ep[prefix + "/chunk_index"], file_index=ep[prefix + "/file_index"])
        targets = [ep[prefix + "/from_timestamp"] + r["timestamp"] for r in selected]
        frames, pts = [], []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                t = float(frame.pts * frame.time_base)
                if abs(t - targets[len(frames)]) <= 0.01 / info["fps"]:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    pts.append(t)
                    if len(frames) == len(targets):
                        break
        if len(frames) != len(targets):
            raise ValueError(f"No timestamp-aligned frames for {key}")
        images[key] = np.stack(frames)
        timestamps[key] = pts
    result = {"main_images": images["observation.images.cam_high"],
              "wrist_images": np.stack([images["observation.images.cam_left_wrist"],
                                         images["observation.images.cam_right_wrist"]], axis=1),
              "states": np.asarray([r["observation.state"] for r in selected], np.float32),
              "task_descriptions": np.asarray([tasks[r["task_index"]] for r in selected])}
    raw_observations(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    args.output.with_suffix(".provenance.json").write_text(json.dumps(
        {"dataset": str(root.resolve()), "episode": 0, "frame_indices": [r["frame_index"] for r in selected],
         "row_timestamps": [r["timestamp"] for r in selected], "decoded_video_timestamps": timestamps,
         "rgb_uint8": True, "purpose": "input parity fixture only; not RL rollout data"}, indent=2) + "\n")
    print("Exported two aligned observations:", args.output)


if __name__ == "__main__":
    main()

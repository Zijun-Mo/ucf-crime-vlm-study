"""Remote CPU full-duration preview for Hydrogen's 100MB native input limit."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/content/annotation_dev_v2');source=root/'Fighting004_x264.mp4';dest=root/'native_video_previews';dest.mkdir(exist_ok=True)
target=dest/source.name
if not target.exists():
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(source),'-vf','fps=8,scale=640:-2','-c:v','libx264','-preset','veryfast','-crf','28','-an','-movflags','+faststart',str(target)],check=True,timeout=240)
assert target.stat().st_size<90*1024**2
meta=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-show_format','-show_streams','-of','json',str(target)]))
duration=float(meta['format']['duration']);assert abs(duration-559.2)<.3
report=dict(name=source.name,kind='full-duration remote native video preview; not contact sheets',original_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    preview_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),preview_bytes=target.stat().st_size,duration_s=duration,
    preview_fps=8.,original_timeline_offset_s=0.,max_width=640,reason='Hydrogen read_file rejects original 121757725-byte video at 100MB cap',original_kept_remote=True)
(dest/'preview_metadata.json').write_text(json.dumps(report,indent=2)+'\n')
cmd=['/content/ucf-bin/rclone','--config','/content/ucf-rclone.conf','copy',str(dest),'gdrive:remaining_20261108_v1/localization_dev_v2/native_video_previews','--transfers','2','--retries','2']
subprocess.run(cmd,check=True,timeout=150);print('NATIVE_FULL_VIDEO_PREVIEW_READY',json.dumps(report),flush=True)

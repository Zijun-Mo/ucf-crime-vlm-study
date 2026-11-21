"""Create downloadable evidence archives and an independent offline website."""
import hashlib,json,zipfile,tarfile,io
from pathlib import Path
HERE=Path(__file__).resolve().parent
DIST=HERE/'dist'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def size(p):return sum(x.stat().st_size for x in p.rglob('*') if x.is_file())
def zip_files(path,files):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(set(files)):
            z.write(p,p.relative_to(DIST))

def main():
    # A self-referential manifest and archive catalogs are deliberately excluded.
    payload=[p for p in DIST.rglob('*') if p.is_file() and p.relative_to(DIST).parts[0] in ['data','assets'] and p.name not in ['files.sha256','downloads.js'] and p.suffix!='.js']
    payload += [DIST/n for n in ['verify_results.py','study_evaluation.py','requirements.txt','README.md']]
    (DIST/'data/files.sha256').write_text(''.join(sha(p)+'  '+p.relative_to(DIST).as_posix()+'\n' for p in sorted(payload)))
    payload.append(DIST/'data/files.sha256')
    out=DIST/'downloads';out.mkdir(exist_ok=True)
    entries=[]
    specs=[
        ('all-results.zip','完整结果资料包','20条件、逐视频及窗口分数、教师与开发结果、图表、历史和CPU复算入口。',payload),
        ('test-results.zip','正式测试与配对统计','20条件的完整窗口分数、逐视频指标、时间元数据、配置来源和配对比较。',[p for p in payload if p.relative_to(DIST).as_posix().startswith(('data/predictions/','data/reports/','data/per_video/','data/statistics/','data/provenance/')) or p.name in ['summary.csv','summary.json','per_video.csv','test_videos.json','comparisons.json','data_dictionary.md','protocol.json','e9_ablation.json','e9_ablation_protocol.json']]),
        ('training-development.zip','训练、教师与开发集','1419份冻结教师分数，九轮开发预测和指标、近似时间标注、科学配置与选模记录。',[p for p in payload if p.relative_to(DIST).as_posix().startswith('data/training/')]),
        ('figures-analysis.zip','图表与案例证据','可直接用于报告的PNG／SVG、固定案例画面、迁移分解和分层数据。',[p for p in payload if p.relative_to(DIST).as_posix().startswith(('assets/','data/analysis/'))]),
        ('historical-results.zip','历史结果附录','早期步长、MIL／DSS结果及报告；协议与当前正式条件不同，单独保留。',[p for p in payload if p.relative_to(DIST).as_posix().startswith('data/history/')]),
    ]
    for filename,title,description,files in specs:
        p=out/filename;zip_files(p,files)
        if p.stat().st_size>=25*1024*1024:raise ValueError(f'Archive exceeds static hosting limit: {p.name}')
        entries.append(dict(title=title,description=description,path='downloads/'+filename,bytes=p.stat().st_size,sha256=sha(p)))
    # Offline contains the actual unpacked data, not duplicate nested archives.
    offline_files=[p for p in DIST.rglob('*') if p.is_file() and p.relative_to(DIST).parts[0]!='downloads' and p.name!='downloads.js' and not any(x in ['__pycache__'] for x in p.parts)]
    offline=out/'offline-website.tar.gz'
    with tarfile.open(offline,'w:gz',compresslevel=9) as archive:
        for p in sorted(offline_files):archive.add(p,arcname=p.relative_to(DIST),recursive=False)
        catalog_bytes=b'window.RESULT_DOWNLOADS=[];'
        info=tarfile.TarInfo('data/downloads.js');info.size=len(catalog_bytes)
        archive.addfile(info,io.BytesIO(catalog_bytes))
    if offline.stat().st_size>=25*1024*1024:raise ValueError('Offline website exceeds static hosting limit')
    entries.append(dict(title='离线网页＋已解压结果',description='tar.gz格式；解压后打开index.html即可阅读，包含全部数据、图表与CPU复算入口，无需联网。',path='downloads/offline-website.tar.gz',bytes=offline.stat().st_size,sha256=sha(offline)))
    catalog=DIST/'data/downloads.js';catalog.write_text('window.RESULT_DOWNLOADS='+json.dumps(entries,ensure_ascii=False,separators=(',',':'))+';')
    checks=dict(files_in_evidence_manifest=len(payload)-1,archives=entries,site_uncompressed_bytes=size(DIST))
    (HERE/'package_acceptance.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    print(json.dumps(checks,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

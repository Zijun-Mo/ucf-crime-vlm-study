"""Archive only small JPEG evidence, never original videos."""
from pathlib import Path
import zipfile
folder=Path('/content/annotation_dev_v2/sheets');target=Path('/content/annotation_review_sheets.zip')
with zipfile.ZipFile(target,'w',zipfile.ZIP_STORED) as archive:
    for path in sorted(folder.rglob('*.jpg')):archive.write(path,path.relative_to(folder))
print('ANNOTATION_JPEG_ARCHIVE',target.stat().st_size,flush=True)

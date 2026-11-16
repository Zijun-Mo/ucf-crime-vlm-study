import numpy as np
import pytest
from openjev_dss.storyboard_context import sheet_plan,make_sheets


@pytest.mark.parametrize('shape',[(320,240),(1920,1080),(240,640),(96,64),(224,224)])
def test_plan_preserves_source_and_bounds_real_token_geometry(shape):
    w,h=shape;p=sheet_plan(w,h,3000,30.)
    assert p['estimated_image_tokens']<=768 and p['pages']<=2
    assert p['image_width']<=w and p['image_height']<=h
    assert abs(p['image_width']/p['image_height']-w/h)<.025
    assert p['frame_indices'][0]==0 and p['frame_indices'][-1]==2999
    assert len(p['frame_indices'])==len(set(p['frame_indices']))
    assert p['canvas_width']%32==p['canvas_height']%32==0


def test_actual_montage_uses_uniform_original_frames(tmp_path):
    import cv2
    path=tmp_path/'source.avi';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),2.,(320,240))
    assert writer.isOpened()
    for i in range(20):writer.write(np.full((240,320,3),i*10,dtype=np.uint8))
    writer.release();images,report=make_sheets(path,tmp_path/'sheets')
    assert len(images)==report['pages']==1
    assert report['frame_indices']==[0,6,13,19]
    assert images[0].getpixel((100,100))[0]<images[0].getpixel((420,100))[0]
    assert len(list((tmp_path/'sheets').glob('*.jpg')))==1

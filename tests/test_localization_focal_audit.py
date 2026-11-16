from openjev_dss.localization_focal_audit import development_metrics


def test_unannotated_abnormal_videos_are_excluded_from_frame_validation():
    records=[{'name':'normal','label':0},{'name':'reviewed','label':1},{'name':'unreviewed','label':1}]
    meta={name:dict(name=name,n_frames=8,fps=1.) for name in ['normal','reviewed','unreviewed']}
    annotations={'reviewed':{**meta['reviewed'],'intervals':[[4,6]]}}
    scores={'normal':[-4.]*8,'reviewed':[-2.,-1.,2.,4.,2.,0.,-2.,-3.],'unreviewed':[1e6]*8}
    result=development_metrics(records,scores,annotations,meta)
    assert result['dev_frame_videos']==2 and result['dev_frame_count']==16
    scores['unreviewed']=[-1e6]*8
    assert development_metrics(records,scores,annotations,meta)==result

"""Bounded decoded-window sequence; exact existing duration_windows pixels."""
from .study_visuals import duration_windows


class StreamingWindows:
    def __init__(self,path,metadata,batch=4):
        self.path=path;self.metadata=metadata;self.batch=batch;self.iterator=None;self.cursor=0

    def __len__(self):return self.metadata['window_count']

    def __getitem__(self,key):
        if not isinstance(key,slice) or key.step not in [None,1]:raise ValueError('Sequential slices only')
        start=key.start or 0
        if start==0:
            self.close()
            self.iterator=duration_windows(self.path,self.metadata['sampled_frames'],list(range(len(self))),4,self.batch)
            self.cursor=0
        if start!=self.cursor:raise ValueError('Non-sequential window request')
        offset,windows=next(self.iterator)
        assert offset==start and len(windows)==min(self.batch,len(self)-start)
        self.cursor+=len(windows)
        if self.cursor==len(self):
            try:next(self.iterator)
            except StopIteration:pass
            else:raise ValueError('Unexpected extra canonical windows')
        return windows

    def close(self):
        if self.iterator is not None:self.iterator.close();self.iterator=None

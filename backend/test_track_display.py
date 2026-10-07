import unittest
import numpy as np
from track_display import mark_track_gaps,copy_gap_metadata,uploaded_times

class TrackDisplayTests(unittest.TestCase):
    def test_time_gap_and_source_gap_preserve_all_coordinates(self):
        points=[dict(x=float(i*10),y=float(i),z=-10.) for i in range(30)]
        times=np.arange(30,dtype=float)*2;times[15:]+=3600
        marked=mark_track_gaps(points,times)
        self.assertEqual([i for i,p in enumerate(marked) if p['gap_before']],[15])
        self.assertEqual([{k:p[k] for k in ('x','y','z')} for p in marked],points)
        # Sampled observations can straddle a short source gap at ordinary cadence.
        sampled=mark_track_gaps(points[:4],[0,20,40,60],[30])
        self.assertEqual([i for i,p in enumerate(sampled) if p['gap_before']],[2])
        geographic=[dict(lat=i,lon=-i) for i in range(30)]
        latlon=copy_gap_metadata(geographic,marked)
        self.assertEqual([p['gap_before'] for p in latlon],[p['gap_before'] for p in marked])

    def test_continuous_sparse_track_is_not_split_by_distance_alone(self):
        points=[dict(x=1000.*i,y=0.) for i in range(30)]
        result=mark_track_gaps(points,np.arange(30)*600.)
        self.assertFalse(any(p['gap_before'] for p in result))

    def test_missing_times_large_jump_and_reversed_time(self):
        points=[dict(x=float(i)+(10000 if i>=15 else 0),y=0.) for i in range(30)]
        self.assertEqual([i for i,p in enumerate(mark_track_gaps(points)) if p['gap_before']],[15])
        result=mark_track_gaps([dict(x=i,y=0) for i in range(4)],[0,1,-1,2])
        self.assertTrue(result[2]['gap_before'])
        self.assertIsNone(uploaded_times([{'Time_UTC':'invalid'}]))
        self.assertEqual(uploaded_times([{'Time_UTC':'2021-01-01T00:00:00Z'},{'Time_UTC':'2021-01-02T00:00:00Z'}]).tolist(),[0.,86400.])

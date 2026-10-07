import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app

class FieldUploadTests(unittest.TestCase):
    def test_gnss_preview_preserves_north_coordinate(self):
        csv='Time_UTC,Latitude_deg,Longitude_deg,Altitude_m\n2021-05-01T00:00:00Z,18.7,116.5,24\n2021-05-01T00:00:02Z,18.71,116.51,24\n'
        r=TestClient(app).post('/api/gnss/parse',files={'gnss_file':('gnss.csv',csv,'text/csv')})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['n_points'],2)
        self.assertGreater(r.json()['trajectory'][1]['y']-r.json()['trajectory'][0]['y'],1000)

    def test_profile_preserves_every_value(self):
        import numpy as np
        values=[(i,1500+np.sin(i/5)*3+i*.002) for i in range(300)]
        csv='Depth (m),SV (m/s)\n'+''.join(f'{z},{c:.12f}\n' for z,c in values)
        r=TestClient(app).post('/api/ssp/parse',files={'svp_file':('s.csv',csv,'text/csv')},data={'smooth':'1','epsilon':'100'})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['n_raw'],300)
        self.assertEqual(r.json()['optimized_points'],[])
        np.testing.assert_allclose([p['sv'] for p in r.json()['raw_points']],[float(f'{c:.12f}') for z,c in values],rtol=0,atol=0)

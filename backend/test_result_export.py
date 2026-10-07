import csv
import io
import json
import unittest
import zipfile
from fastapi.testclient import TestClient
from main import app, TASK_STORE, _json_safe
from result_export import make_archive
from joint_simulation import run_survey

class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        body=dict(array_points=[[400,400,-1500],[-400,400,-1510],[-400,-400,-1520],[400,-400,-1530]],
                  modes=['figure_eight'],size_ratios=[.7,1.,1.3],ship_speed_kn=3,noise=0,
                  gnss_noise_std=0,temporal_sigma_mps=0,sv_grad_e=0,sv_grad_n=0,
                  sample_freq_hz=.05,time_budget_min_h=1,time_budget_max_h=5,gdop_threshold=100)
        cls.saved=dict(kind='simulation')
        result=run_survey(body,export_sink=lambda rows:cls.saved.update(rows=rows))
        cls.saved['result']=result
        TASK_STORE['export-test']=dict(status='completed',result=result,_export=cls.saved)

    @classmethod
    def tearDownClass(cls):TASK_STORE.pop('export-test',None)

    def archive(self,options):
        with TestClient(app) as client:
            response=client.post('/api/export/export-test',json=options)
        self.assertEqual(response.status_code,200,response.text if response.status_code!=200 else '')
        return zipfile.ZipFile(io.BytesIO(response.content))

    def test_all_preserves_saved_observations_and_failed_candidates(self):
        z=self.archive(dict(scope='all',parts=['trajectory','observations','residuals']))
        data=json.loads(z.read('results.json'))
        self.assertEqual(len(data['results']),len(self.saved['rows']))
        for original,exported in zip(self.saved['rows'],data['results']):
            self.assertEqual(original['trajectory'],exported['trajectory'])
            self.assertEqual(original.get('_observations',[]),exported['observations'])
            self.assertEqual([r['residual_ms'] for r in exported['residuals']],original.get('residuals_ms',[]))
        with TestClient(app) as client:
            progress=client.get('/api/progress/export-test').json()
        self.assertNotIn('_export',progress)

    def test_top10_respects_selected_budget_and_gdop(self):
        cap=self.saved['result']['budget_values_h'][4]
        z=self.archive(dict(scope='top10',budget_h=cap,parts=[]))
        data=json.loads(z.read('results.json'))['results']
        self.assertLessEqual(len(data),10)
        self.assertTrue(all(r['valid'] and r['duration_h']<=cap+1e-10 for r in data))
        self.assertEqual([r['c3_m2'] for r in data],sorted(r['c3_m2'] for r in data))
        self.assertTrue(all('trajectory' not in r and 'observations' not in r for r in data))

    def test_custom_exact_ids_and_invalid_id_rejected(self):
        wanted=self.saved['rows'][-1]['candidate_id']
        z=self.archive(dict(scope='custom',ids=[wanted],parts=['observations']))
        data=json.loads(z.read('results.json'))['results']
        self.assertEqual([r['candidate_id'] for r in data],[wanted])
        with TestClient(app) as client:
            self.assertEqual(client.post('/api/export/export-test',json=dict(scope='custom',ids=['wrong-run'])).status_code,422)

    def test_field_selection_keeps_residual_association_and_originals(self):
        rows=[dict(beacon_id=i,status='success',rmse_time_ms=float(4-i),
                   observations=[dict(beacon_id=i,time_s=100+i)],residuals_ms=[i*.1],
                   outlier_mask=[False],trajectory=[dict(x=i,y=i+1,z=-20)]) for i in [2,3]]
        saved=dict(kind='field',rows=rows,result={},config={'solver_type':'7param'},raw={'delay.csv':b'original,unchanged\r\n2,3\r\n'})
        z=zipfile.ZipFile(io.BytesIO(make_archive(saved,dict(scope='custom',ids=['3'],parts=['residuals','raw']),_json_safe)))
        data=json.loads(z.read('results.json'))['results']
        self.assertEqual(len(data),1)
        self.assertEqual(data[0]['residuals'][0]['beacon_id'],3)
        self.assertEqual(data[0]['residuals'][0]['time_s'],103)
        self.assertEqual(z.read('original_inputs/delay.csv'),saved['raw']['delay.csv'])
        z=zipfile.ZipFile(io.BytesIO(make_archive(saved,dict(scope='top10',parts=[]),_json_safe)))
        summary=list(csv.DictReader(io.StringIO(z.read('summary.csv').decode('utf-8-sig'))))
        self.assertEqual([r['beacon_id'] for r in summary],['3','2'])

if __name__=='__main__':unittest.main()

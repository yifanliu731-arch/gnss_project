<!--
  RealDataMode.vue - Field data组件 (v2.3)
  左栏（宽 md="17"）：结果展示区
  右栏（窄 md="7"）：Input files区
  上传文件时自动实时更新图表
-->
<template>
  <div class="real-data-mode"><p class="field-scope">{{ t("Joint calibration estimates all selected beacon coordinates with shared acoustic parameters. Single-beacon fitting uses the same propagation and estimation model independently for each beacon.") }}</p>
    <el-row :gutter="20">
      <!-- ==================== 左栏：结果展示区 ==================== -->
      <el-col :xs="24" :sm="24" :md="17" :lg="17" :xl="17">
        <!-- Sound-speed profile图 -->
        <el-card class="panel-card" shadow="hover">
          <template #header>
            <div class="panel-header">
              <el-icon><DataAnalysis /></el-icon>
              <span>{{ t("Sound-speed profile") }}</span>
              <el-tag v-if="sspN > 0" size="small" type="success" style="margin-left:8px">{{ t(sspN) }} {{ t("layers") }}</el-tag>
            </div>
          </template>
          <SSPProfileChart :raw-points="sspRawPoints" :transducer-depth="Math.abs(leverArm[2])" />
        </el-card>

        <!-- Survey track图表 (ENU) -->
        <el-card class="panel-card" shadow="hover">
          <template #header>
            <div class="panel-header">
              <el-icon><DataAnalysis /></el-icon>
              <span>{{ t("Survey track (ENU)") }}</span>
              <span v-if="refLat !== null && refLon !== null" style="margin-left:auto;font-size:12px;color:#909399">{{ t("Reference origin:") }} {{ t(refLat.toFixed(1)) }}°, {{ t(refLon.toFixed(1)) }}°</span>
            </div>
          </template>
          <el-radio-group v-model="trackDisplayMode" size="small" :aria-label="t('Track display')" class="track-display-controls">
            <el-radio-button value="smooth">{{ t('Smooth curve') }}</el-radio-button>
            <el-radio-button value="scatter">{{ t('Points only') }}</el-radio-button>
          </el-radio-group>
          <TrajectoryChart v-if="hasChartData" :trajectory-data="trajectoryData" :display-mode="trackDisplayMode" />
          <p v-if="hasChartData" class="field-scope">{{ t(trackDisplayMode==='smooth'?'Smoothing is for display only. Pale dashed links indicate gaps, not observed travel.':'Points are shown without connecting lines. Display mode does not change the fit.') }}</p>
          <div v-else class="chart-placeholder"><el-icon :size="40" color="#c0c4cc"><TrendCharts /></el-icon><p>{{ t("Upload a GNSS position file to view the track") }}</p></div>
        </el-card>

        <!-- Survey track图表 (经Latitude) -->
        <el-card class="panel-card" shadow="hover">
          <template #header><div class="panel-header"><el-icon><DataAnalysis /></el-icon><span>{{ t("Survey track (geographic)") }}</span></div></template>
          <el-radio-group v-model="trackDisplayMode" size="small" :aria-label="t('Geographic track display')" class="track-display-controls">
            <el-radio-button value="smooth">{{ t('Smooth curve') }}</el-radio-button>
            <el-radio-button value="scatter">{{ t('Points only') }}</el-radio-button>
          </el-radio-group>
          <TrajectoryLatLonChart v-if="latlonTrajectoryData.length > 0" :trajectory-data="latlonTrajectoryData" :ref-lat="refLat ?? 0" :ref-lon="refLon ?? 0" :display-mode="trackDisplayMode" />
          <div v-else class="chart-placeholder"><el-icon :size="40" color="#c0c4cc"><TrendCharts /></el-icon><p>{{ t("Upload a GNSS position file to view the track") }}</p></div>
        </el-card>

        <!-- Travel-time residuals -->
        <el-card class="panel-card" shadow="hover">
          <template #header><div class="panel-header"><el-icon><DataAnalysis /></el-icon><span>{{ t("Travel-time residuals") }}</span></div></template>
          <RealResidualChart v-if="residualData.length > 0" :data="residualData" :outlier-mask="outlierMask" :title="t(&quot;Travel-time residuals&quot;)" :y-axis-name="t(&quot;Residual (ms)&quot;)" unit="ms" />
          <p v-if="residualData.length && ['joint_map','single_map'].includes(result.solverType ?? '')" class="field-scope">{{ t('Residual = observed time − fitted prediction, in ms. The prediction includes refracted travel time, fitted sound-speed offset, effective delays and drift. Residuals are not coordinate errors; no residual clipping is applied in this mode.') }}</p>
          <div v-else class="chart-placeholder"><el-icon :size="40" color="#c0c4cc"><TrendCharts /></el-icon><p>{{ t("Upload files and select \"Process field data\" to view residuals") }}</p></div>
        </el-card>

        <!-- Equivalent range residuals -->
        <el-card class="panel-card" shadow="hover">
          <template #header><div class="panel-header"><el-icon><DataAnalysis /></el-icon><span>{{ t("Equivalent range residuals") }}</span></div></template>
          <RealResidualChart v-if="distanceResidualData.length > 0" :data="distanceResidualData" :outlier-mask="outlierMask" :title="t(&quot;Equivalent range residuals&quot;)" :y-axis-name="t(&quot;Residual (m)&quot;)" unit="m" />
          <p v-if="distanceResidualData.length" class="field-scope">{{ t('Equivalent range residual = travel-time residual (s) × equivalent sound speed. This is not a position error.') }}</p>
          <div v-else class="chart-placeholder"><el-icon :size="40" color="#c0c4cc"><TrendCharts /></el-icon><p>{{ t("Equivalent range residuals appear after TOA adjustment") }}</p></div>
        </el-card>

        <!-- Estimation results -->
        <el-card class="panel-card result-card" shadow="hover">
          <template #header>
            <div class="panel-header">
              <el-icon><List /></el-icon><span>{{ t("Estimation results") }}</span>
              <el-tag v-if="result.solverType" size="small" type="info" style="margin-left:12px">{{ t(solverLabel(result.solverType)) }}</el-tag>
            </div>
          </template>
          <template v-if="jointSolution">
            <p class="field-scope">{{ t('All beacon rows below belong to one joint fit. Internal standard deviations are conditional precision, not errors against coordinate truth.') }}</p>
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item :label="t('Joint parameters')">{{ jointSolution.joint_parameter_count }} (4N + 2)</el-descriptions-item>
              <el-descriptions-item :label="t('Relative-delay reference beacon')">B{{ jointSolution.reference_beacon_id }}</el-descriptions-item>
              <el-descriptions-item :label="t('Shared sound-speed offset (m/s)')">{{ displayNumber(jointSolution.sound_speed_offset_mps) }}</el-descriptions-item>
              <el-descriptions-item :label="t('Common effective delay (ms)')">{{ displayNumber(jointSolution.common_delay_s*1000) }}</el-descriptions-item>
              <el-descriptions-item :label="t('Delay drift (ms/h)')">{{ displayNumber(jointSolution.delay_rate_s_per_hour*1000) }}</el-descriptions-item>
              <el-descriptions-item :label="t('Effective residual degrees of freedom')">{{ displayNumber(jointSolution.effective_residual_dof,2) }}</el-descriptions-item>
            </el-descriptions>
          </template>
          <el-table v-if="beaconResultList.length" :data="beaconResultList" size="small" style="margin-bottom:12px" :aria-label="t('Beacon estimation results')">
            <el-table-column prop="beacon_id" :label="t('Beacon ID')" width="80"/>
            <el-table-column :label="t('E (m)')" min-width="100"><template #default="{row}">{{ displayNumber(row.x,3) }}</template></el-table-column>
            <el-table-column :label="t('N (m)')" min-width="100"><template #default="{row}">{{ displayNumber(row.y,3) }}</template></el-table-column>
            <el-table-column :label="t('U (m)')" min-width="100"><template #default="{row}">{{ displayNumber(row.z,3) }}</template></el-table-column>
            <el-table-column :label="t('RMS (ms)')" min-width="90"><template #default="{row}">{{ displayNumber(row.rmse_time_ms,4) }}</template></el-table-column>
            <el-table-column :label="t('σE / σN / σU (m)')" min-width="185"><template #default="{row}">{{ displayNumber(row.sigma_e_m) }} / {{ displayNumber(row.sigma_n_m) }} / {{ displayNumber(row.sigma_u_m) }}</template></el-table-column>
            <el-table-column prop="n_clean" :label="t('Observations')" width="100"/>
          </el-table>
          <p v-if="beaconResultList.length && !jointSolution" class="field-scope">{{ t('The details below show the first successfully fitted beacon. All selected beacons are listed in the table and can be exported.') }}</p>
          <el-descriptions v-if="!jointSolution" :column="2" border size="default">
            <el-descriptions-item :label="t(&quot;X (m)&quot;)" label-align="center">{{ t(result.x ?? '-') }}</el-descriptions-item>
            <el-descriptions-item :label="t(&quot;Y (m)&quot;)" label-align="center">{{ t(result.y ?? '-') }}</el-descriptions-item>
            <el-descriptions-item :label="t(&quot;Z (m)&quot;)" label-align="center">{{ t(result.z ?? '-') }}</el-descriptions-item>
            <el-descriptions-item :label="t(&quot;RMS (ms)&quot;)" label-align="center">{{ t(result.rmse ?? '-') }}</el-descriptions-item>
            <el-descriptions-item :label="t(&quot;Range RMS (m)&quot;)" label-align="center">{{ t(result.rmseDistance ?? '-') }}</el-descriptions-item>
            <el-descriptions-item :label="t(&quot;Outlier rejection rate&quot;)" label-align="center" :span="2">{{ t(result.outlierRate ?? '-') }}</el-descriptions-item>
            <el-descriptions-item :label="t(&quot;Equivalent sound speed C_EQ (m/s)&quot;)" label-align="center">{{ t(result.cEq ?? '-') }}</el-descriptions-item>
            <el-descriptions-item :label="t(&quot;σE / σN / σU (m)&quot;)" label-align="center">{{ t(result.sigmaE ?? '-') }} / {{ t(result.sigmaN ?? '-') }} / {{ t(result.sigmaU ?? '-') }}</el-descriptions-item>
            <template v-if="result.systemClockBiasMs != null">
              <el-descriptions-item :label="t(&quot;Primary-session clock offset B (ms)&quot;)" label-align="center">{{ t(result.systemClockBiasMs) }}</el-descriptions-item>
              <el-descriptions-item :label="t(&quot;Clock drift D1 (s/s)&quot;)" label-align="center">{{ t(result.clockDriftSs) }}</el-descriptions-item>
              <el-descriptions-item :label="t(&quot;Clock drift D2 (s/s²)&quot;)" label-align="center">{{ t(result.clockDrift2 ?? '-') }}</el-descriptions-item>
              <el-descriptions-item :label="t(&quot;Zenith delay k (ms/km²)&quot;)" label-align="center">{{ t(result.zenithK) }}</el-descriptions-item>
              <el-descriptions-item v-if="result.svGradE != null" :label="t(&quot;East gradient (m/s/km)&quot;)" label-align="center">{{ t(result.svGradE) }}</el-descriptions-item>
              <el-descriptions-item v-if="result.svGradN != null" :label="t(&quot;North gradient (m/s/km)&quot;)" label-align="center">{{ t(result.svGradN) }}</el-descriptions-item>
              <el-descriptions-item v-if="result.abic != null" :label="t(&quot;ABIC / λ&quot;)" label-align="center">{{ t(result.abic) }} / {{ t(result.abicLambda) }}</el-descriptions-item>
            </template>
            <template v-if="result.solverType === 'eskf_21param'">
              <el-descriptions-item :label="t(&quot;Estimated sound-speed offset (m/s)&quot;)" label-align="center">{{ t(result.svBiasMps ?? '-') }}</el-descriptions-item>
              <el-descriptions-item :label="t(&quot;Estimated fixed delay (ms)&quot;)" label-align="center">{{ t(result.sysDelayMs ?? '-') }}</el-descriptions-item>
              <el-descriptions-item :label="t(&quot;Time synchronisation delay (ms)&quot;)" label-align="center">{{ t(result.timeSyncDelayMs ?? '-') }}</el-descriptions-item>
              <el-descriptions-item :label="t(&quot;Lever arm Z (m)&quot;)" label-align="center">{{ t(result.leverArmZ ?? '-') }}</el-descriptions-item>
            </template>
          </el-descriptions>
        </el-card>
        <ResultExportPanel v-if="beaconResultList.length && !isProcessing" :key="runTaskId" :task-id="runTaskId" kind="field" :rows="beaconResultList"/>
      </el-col>

      <!-- ==================== 右栏：Input files区 ==================== -->
      <el-col :xs="24" :sm="24" :md="7" :lg="7" :xl="7">
        <el-card class="panel-card" shadow="hover">
          <template #header><div class="panel-header"><el-icon><Upload /></el-icon><span>{{ t("Input files") }}</span></div></template>
          <div class="upload-section">
            <div class="upload-item">
              <div class="upload-label"><el-tag type="primary" size="small">{{ t("File 1") }}</el-tag><span>{{ t("GNSS positions (.csv)") }}</span></div>
              <el-upload ref="gnssUploadRef" v-model:file-list="gnssFileList" class="upload-box" drag action="#" :auto-upload="false" :limit="1" accept=".csv" :on-exceed="handleExceed" :on-change="handleGnssFileChange">
                <el-icon class="upload-icon"><UploadFilled /></el-icon>
                <div class="upload-text">{{ t("Drop a file or") }} <em>{{ t("browse") }}</em></div>
                 <template #tip>
                   <div class="upload-tip-wrap">
                     <code>Time_UTC, Latitude_deg, Longitude_deg,</code>
                     <code>Altitude_m, Pitch_deg, Roll_deg, Heading_deg</code>
                     <div style="margin-top:4px;border-top:1px dashed #dcdfe6;padding-top:4px;text-align:left">
                       <p style="font-size:10px;color:#909399;margin:0 0 4px;font-weight:500">{{ t("🛰️ Optional inertial navigation columns (INS):") }}</p>
                       <code style="font-size:10px">gyro_x, gyro_y, gyro_z</code>
                       <code style="font-size:10px">acc_x, acc_y, acc_z</code>
                       <p style="font-size:9px;color:#c0c4cc;margin:2px 0 0">{{ t("INS columns enable loosely coupled ESKF navigation") }}</p>
                     </div>
                   </div>
                 </template>
              </el-upload>
            </div>
            <div class="upload-item">
              <div class="upload-label"><el-tag type="success" size="small">{{ t("File 2") }}</el-tag><span>{{ t("Travel-time data (.csv)") }}</span></div>
              <el-upload ref="delayUploadRef" v-model:file-list="delayFileList" class="upload-box" drag action="#" :auto-upload="false" :limit="1" accept=".csv" :on-exceed="handleExceed" :on-change="handleDelayFileChange">
                <el-icon class="upload-icon"><UploadFilled /></el-icon>
                <div class="upload-text">{{ t("Drop a file or") }} <em>{{ t("browse") }}</em></div>
                <template #tip><div class="upload-tip-wrap"><code>Time_UTC, Beacon_ID, Delay_s</code></div></template>
              </el-upload>
            </div>
            <div class="upload-item">
              <div class="upload-label"><el-tag type="warning" size="small">{{ t("File 3") }}</el-tag><span>{{ t("SSP Sound-speed profile (.csv)") }}</span></div>
              <el-upload ref="svpUploadRef" v-model:file-list="svpFileList" class="upload-box" drag action="#" :auto-upload="false" :limit="1" accept=".csv" :on-exceed="handleExceed" :on-change="handleSvpFileChange">
                <el-icon class="upload-icon"><UploadFilled /></el-icon>
                <div class="upload-text">{{ t("Drop a file or") }} <em>{{ t("browse") }}</em></div>
                <template #tip><div class="upload-tip-wrap"><code>Depth_m, Velocity_mps</code></div></template>
              </el-upload>
            </div>
          </div>
          <p class="field-scope">{{ t("Uploaded profile values are used without smoothing or decimation.") }}</p>
          <el-divider />
          <p style="font-weight:600;margin:0 0 4px;font-size:12px">{{ t("Beacon IDs") }}</p>
          <el-input v-model="beaconIds" :placeholder="t(&quot;e.g. 5, 6, 7&quot;)" size="small" style="margin-bottom:8px" />
          <p class="field-scope">{{ t('Separate IDs with commas or spaces; leave blank to use all IDs found in the uploaded file.') }}</p>
          <el-button v-if="detectedIds.length" size="small" @click="beaconIds=detectedIds.join(',')">{{ t('Use all detected IDs') }}: {{ detectedIds.join(', ') }}</el-button>
          <p style="font-weight:600;margin:10px 0 4px;font-size:12px">{{ t("Solver") }}</p>
          <el-radio-group v-model="solverType" size="small" style="width:100%;margin-bottom:8px">
            <el-radio-button value="joint_map">{{ t('Joint array calibration') }}</el-radio-button>
            <el-radio-button value="single_map">{{ t('Single-beacon fitting') }}</el-radio-button>
            <el-radio-button value="7param">{{ t("TOA: fixed k") }}</el-radio-button>
            <el-radio-button value="9param">{{ t("TOA + gradients ABIC") }}</el-radio-button>
            <el-radio-button value="eskf_21param">{{ t("21-state ESKF calibration") }}</el-radio-button>
          </el-radio-group>
          <div v-if="isSharedModel">
            <p class="field-scope">{{ t('Joint and single-beacon modes share free 3D coordinates, curved-ray propagation and acoustic priors. Single mode fits each selected beacon separately.') }}</p>
            <p>{{ t('Timing noise σ (ms)') }}</p><el-input-number v-model="jointTimingNoiseMs" :min=".001" :max="100" :step=".1" :precision="3" size="small"/>
            <p>{{ t('GNSS σ (m)') }}</p><el-input-number v-model="jointGnssNoiseM" :min="0" :max="10" :step=".01" :precision="3" size="small"/>
          </div>
          <p style="font-weight:600;margin:10px 0 4px;font-size:12px">{{ t("Lever arm: right / forward / up (m)") }}</p>
          <p class="field-scope">{{ t('Lever arm is the transducer position relative to the GNSS antenna, expressed in the vessel local frame: right, forward, up. A downward offset is negative; attitude rotates this vector into ENU.') }}</p>
          <el-input v-model="leverArm[0]" :placeholder="t(&quot;Right (m)&quot;)" size="small" style="margin-bottom:4px"><template #prepend>{{ t("Right") }}</template></el-input>
          <el-input v-model="leverArm[1]" :placeholder="t(&quot;Forward (m)&quot;)" size="small" style="margin-bottom:4px"><template #prepend>{{ t("Forward") }}</template></el-input>
          <el-input v-model="leverArm[2]" :placeholder="t(&quot;Up (m)&quot;)" size="small" style="margin-bottom:4px"><template #prepend>{{ t("Up") }}</template></el-input>
          <div v-if="isProcessing" style="margin-top:8px">
            <el-progress :percentage="progress" :text-inside="true" :stroke-width="18" />
            <p style="font-size:11px;color:#909399;margin:4px 0 0;text-align:center">{{ t(progressMessage || 'Processing...') }}</p>
          </div>
          <div style="display:flex;justify-content:center;margin-top:8px">
            <el-button type="primary" :loading="isProcessing" :icon="VideoPlay" :disabled="!isAllFilesReady" size="large" style="min-width:160px;padding:12px 28px;font-size:15px" @click="handleStartProcessing">
              {{ t(isProcessing ? 'Processing...' : 'Process field data') }}
            </el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { t, locale, setLocalizedOption } from "../i18n"
import { computed, reactive, ref } from 'vue'
import { Upload, UploadFilled, VideoPlay, DataAnalysis, TrendCharts, List } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import type { UploadFile, UploadFiles, UploadInstance } from 'element-plus'
import { apiClient } from '../api'
import TrajectoryChart from './TrajectoryChart.vue'
import type { TrackPoint } from '../trackSegments'
import TrajectoryLatLonChart from './TrajectoryLatLonChart.vue'
import RealResidualChart from './RealResidualChart.vue'
import SSPProfileChart from './SSPProfileChart.vue'
import ResultExportPanel from './ResultExportPanel.vue'

const gnssFileList = ref<UploadFile[]>([])
const delayFileList = ref<UploadFile[]>([])
const svpFileList = ref<UploadFile[]>([])
const gnssUploadRef = ref<UploadInstance>()
const delayUploadRef = ref<UploadInstance>()
const svpUploadRef = ref<UploadInstance>()
const isProcessing = ref(false)
const progress = ref(0)
const progressMessage = ref('')
const leverArm = reactive([1.585, 6.716, -29.05])
const solverType = ref("joint_map")
const beaconIds = ref("")
const detectedIds = ref<number[]>([])
const jointTimingNoiseMs=ref(.5), jointGnssNoiseM=ref(.05)
const jointSolution=ref<any>(null)
const isSharedModel=computed(()=>['joint_map','single_map'].includes(solverType.value))
const displayNumber=(value:any,digits=4)=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(digits):'-'


interface BeaconResult { beacon_id: number; status: string; x: number|null; y: number|null; z: number|null; rmse_time_ms: number|null; outlier_rate: string|null; n_total: number|null; n_clean: number|null; system_clock_bias_ms?: number|null; clock_drift_ss?: number|null; hardware_time_lag_ms?: number|null; zenith_k_ms_km2?: number|null; lever_arm_x_starboard_m?: number|null; lever_arm_y_forward_m?: number|null }
const beaconResultList = ref<BeaconResult[]>([])
const runTaskId=ref('')

const result = reactive({
  x: null as number|null, y: null as number|null, z: null as number|null,
  rmse: null as number|null, rmseDistance: null as number|null, outlierRate: null as string|null,
  systemClockBiasMs: null as number|null, clockDriftSs: null as number|null, clockDrift2: null as number|null,
  zenithK: null as number|null, cEq: null as number|null, sigmaE: null as number|null, sigmaN: null as number|null, sigmaU: null as number|null, solverType: null as string|null,
  svGradE: null as number|null, svGradN: null as number|null, abic: null as number|null, abicLambda: null as number|null,
  svBiasMps: null as number|null, sysDelayMs: null as number|null, timeSyncDelayMs: null as number|null, leverArmZ: null as number|null,
})
function solverLabel(t: string|null): string { return t==='joint_map'?'Joint array calibration':t==='single_map'?'Single-beacon fitting':t==='7param'?'TOA: fixed k':t==='9param'?'TOA + gradients':t==='abic_grad'?'TOA + gradients ABIC':t==='eskf_21param'?'21-state calibration':t==='simple_ls'?'Basic LS':t??'?' }

const trackDisplayMode = ref<'smooth'|'scatter'>('smooth')
const trajectoryData = ref<TrackPoint[]>([])
type ResidualPoint = {epoch:number;value:number;beacon_id?:number;source_time?:string}
const residualData = ref<ResidualPoint[]>([])
const distanceResidualData = ref<ResidualPoint[]>([])
const outlierMask = ref<boolean[]>([])
const refLat = ref<number|null>(null); const refLon = ref<number|null>(null)
const latlonTrajectoryData = ref<Array<{lat:number;lon:number;gap_before?:boolean}>>([])
const isAllFilesReady = computed(() => gnssFileList.value.length>0 && delayFileList.value.length>0 && svpFileList.value.length>0)
const hasChartData = computed(() => trajectoryData.value.length>0)

interface SSPPoint { depth: number; sv: number }
const sspRawPoints = ref<SSPPoint[]>([])
const sspN = ref(0)

function handleExceed() { localizedMessage.warning('Remove the existing file before uploading another') }

/** GNSS 上传 → 实时解析轨迹并显示图表 */
async function handleGnssFileChange(file: UploadFile, _files: UploadFiles) {
  console.log('[RealData] GNSS file changed:', file.name, file.uid)
  const raw = file?.raw as File|undefined
  if (!raw) { console.warn('[RealData] GNSS: no raw file'); return }
  const fd = new FormData(); fd.append('gnss_file', raw)
  try {
    const r = await apiClient.post('/api/gnss/parse', fd, { headers: {'Content-Type':'multipart/form-data'} })
    console.log('[RealData] GNSS parse response:', r.data)
    if (r.data?.status === 'ok') {
      trajectoryData.value = r.data.trajectory ?? []
      latlonTrajectoryData.value = r.data.trajectory_latlon ?? []
      refLat.value = r.data.ref_lat ?? null
      refLon.value = r.data.ref_lon ?? null
      localizedMessage.success(`GNSS track loaded: ${r.data.n_points ?? '?'} positions`)
    } else {
      localizedMessage.error('Unexpected GNSS parsing response')
    }
  } catch(e: any) {
    const msg = e?.response?.data?.detail ?? e.message
    console.error('[RealData] GNSS parse failed:', msg)
    localizedMessage.error(`GNSS parsing failed: ${msg}`)
  }
}

/** SVP 上传 → 实时解析Sound-speed profile并显示图表 */
let sspRequestId = 0
async function handleSvpFileChange(file: UploadFile, _files: UploadFiles) {
  const requestId = ++sspRequestId
  console.log('[RealData] SVP file changed:', file.name, file.uid)
  const raw = file?.raw as File|undefined
  if (!raw) { console.warn('[RealData] SVP: no raw file'); return }
  const fd = new FormData(); fd.append('svp_file', raw)
  try {
    const r = await apiClient.post('/api/ssp/parse', fd, { headers: {'Content-Type':'multipart/form-data'} })
    if (requestId !== sspRequestId) return
    console.log('[RealData] SSP parse response:', r.data)
    if (r.data?.status === 'ok') {
      sspRawPoints.value = r.data.raw_points ?? []
      sspN.value = r.data.n_raw ?? 0
      localizedMessage.success(`SSP loaded: ${sspN.value} layers`)
    } else {
      localizedMessage.error('Unexpected SSP parsing response')
    }
  } catch(e: any) {
    const msg = e?.response?.data?.detail ?? e.message
    console.error('[RealData] SSP parse failed:', msg)
    localizedMessage.error(`SSP parsing failed: ${msg}`)
  }
}


/** Identify every beacon in the travel-time file without changing observations. */
let delayRequestId=0
async function handleDelayFileChange(file: UploadFile, _files: UploadFiles) {
  const requestId=++delayRequestId; const raw=file?.raw
  if(!raw)return
  detectedIds.value=[];beaconIds.value=''
  const fd=new FormData();fd.append('delay_file',raw)
  try {
    const response=await apiClient.post('/api/delay/parse',fd)
    if(requestId!==delayRequestId)return
    detectedIds.value=response.data.beacon_ids??[]
    beaconIds.value=detectedIds.value.join(',')
    if(detectedIds.value.length)localizedMessage.success(t('Detected beacon IDs')+': '+detectedIds.value.join(', '))
  }catch(e:any){localizedMessage.error(t('Travel-time parsing failed')+': '+(e?.response?.data?.detail??e.message))}
}

async function handleStartProcessing() {
  if (!isAllFilesReady.value) return
  jointSolution.value=null; beaconResultList.value=[]; runTaskId.value=''
  isProcessing.value = true; progress.value = 0; progressMessage.value = 'Submitting task...'
  const fd = new FormData()
  const g = gnssFileList.value[0]?.raw; const d = delayFileList.value[0]?.raw; const s = svpFileList.value[0]?.raw
  if (g) fd.append('gnss_file', g); if (d) fd.append('delay_file', d); if (s) fd.append('svp_file', s)
  fd.append('lever_dx', String(leverArm[0])); fd.append('lever_dy', String(leverArm[1])); fd.append('lever_dz', String(leverArm[2]))
  fd.append('solver_type', solverType.value); fd.append('beacon_ids', beaconIds.value)
  fd.append('joint_noise_ms',String(jointTimingNoiseMs.value));fd.append('joint_gnss_m',String(jointGnssNoiseM.value))
  console.log('[RealDataMode] 发送前参数:')
  console.log('  lever_arm (dx,dy,dz) =', JSON.stringify([...leverArm]))
  console.log('  solver_type =', solverType.value, '| beacon_ids =', beaconIds.value)
  console.log('  files =', gnssFileList.value[0]?.name, '/', delayFileList.value[0]?.name, '/', svpFileList.value[0]?.name)
  try {
    const sr = await apiClient.post('/api/calculate', fd)
    const tid = sr.data?.task_id; if (!tid) throw new Error('Missing task ID'); runTaskId.value=tid
    let done = false; let a = 0
    while (!done && a < 600) { await new Promise(r => setTimeout(r, 1000)); a++; const pr = await apiClient.get(`/api/progress/${tid}`); progress.value = pr.data.progress ?? 0; progressMessage.value = pr.data.message ?? ''; if (pr.data.status === 'completed') done = true; else if (pr.data.status === 'failed') throw new Error(pr.data.error ?? 'Task failed') }
    if (!done) throw new Error('Timed out')
    const fr = await apiClient.get(`/api/progress/${tid}`); const data = fr.data?.result
    if (!data) throw new Error('Empty result')
    jointSolution.value=data.joint_solution??null
    if (data.beacon_results?.length) {
      beaconResultList.value = data.beacon_results
      const f = data.beacon_results.find((row:any)=>row.status==='success')
      if (f) Object.assign(result, { x: f.x, y: f.y, z: f.z, rmse: f.rmse_time_ms, rmseDistance: f.rmse_distance_m ?? null, outlierRate: f.outlier_rate, systemClockBiasMs: f.system_clock_bias_ms ?? null, clockDriftSs: f.clock_drift_ss ?? null, clockDrift2: f.clock_drift2_ss2 ?? null, zenithK: f.zenith_k_ms_km2 ?? null, svGradE: f.sv_grad_e_mps_per_km ?? null, svGradN: f.sv_grad_n_mps_per_km ?? null, abic: f.abic ?? null, abicLambda: f.abic_lambda ?? null, cEq: f.c_eq_mps ?? data.c_eq_mps ?? null, sigmaE: f.sigma_e_m ?? null, sigmaN: f.sigma_n_m ?? null, sigmaU: f.sigma_u_m ?? null, svBiasMps: f.sound_speed_bias_mps ?? null, sysDelayMs: f.system_delay_ms ?? null, timeSyncDelayMs: f.time_sync_delay_ms ?? null, leverArmZ: f.lever_arm_z_down_m ?? null, solverType: f.solver_type ?? data.solver_type ?? 'simple_ls' })
    } else {
      beaconResultList.value = []
      Object.assign(result, { x: data.result_x, y: data.result_y, z: data.result_z, rmse: data.rmse_time_ms ?? data.rmse ?? null, rmseDistance: data.rmse_distance_m ?? null, outlierRate: data.outlier_rate, systemClockBiasMs: data.system_clock_bias_ms ?? null, clockDriftSs: data.clock_drift_ss ?? null, clockDrift2: data.clock_drift2_ss2 ?? null, zenithK: data.zenith_k_ms_km2 ?? null, cEq: data.c_eq_mps ?? null, sigmaE: data.sigma_e_m ?? null, sigmaN: data.sigma_n_m ?? null, sigmaU: data.sigma_u_m ?? null, svBiasMps: data.sound_speed_bias_mps ?? null, sysDelayMs: data.system_delay_ms ?? null, timeSyncDelayMs: data.time_sync_delay_ms ?? null, leverArmZ: data.lever_arm_z_down_m ?? null, solverType: data.solver_type ?? 'simple_ls' })
    }
    if (data.trajectory) trajectoryData.value = data.trajectory
    if (data.residuals_ms) residualData.value = data.residuals_ms.map((v: number, i: number) => ({ ...data.residual_observations?.[i], epoch: i + 1, value: v }))
    if (data.residuals_m) distanceResidualData.value = data.residuals_m.map((v: number, i: number) => ({ ...data.residual_observations?.[i], epoch: i + 1, value: v }))
    if (data.outlier_mask) outlierMask.value = data.outlier_mask
    refLat.value = data.ref_lat ?? null; refLon.value = data.ref_lon ?? null
    if (data.trajectory_latlon) latlonTrajectoryData.value = data.trajectory_latlon
    localizedMessage.success('Field processing complete')
  } catch (e: any) {
    const msg = e?.response?.data?.detail ?? e.message
    console.error('[RealData] Processing failed:', msg)
    localizedMessage.error(`Field processing failed: ${msg}`)
    jointSolution.value=null; trajectoryData.value = []; residualData.value = []; distanceResidualData.value = []; outlierMask.value = []; latlonTrajectoryData.value = []; refLat.value = null; refLon.value = null; beaconResultList.value = []
    Object.assign(result, { x: null, y: null, z: null, rmse: null, rmseDistance: null, outlierRate: null, systemClockBiasMs: null, clockDriftSs: null, clockDrift2: null, zenithK: null, svGradE: null, svGradN: null, abic: null, abicLambda: null, cEq: null, sigmaE: null, sigmaN: null, sigmaU: null, svBiasMps: null, sysDelayMs: null, timeSyncDelayMs: null, leverArmZ: null, solverType: null })
  } finally { isProcessing.value = false; progress.value = 0; progressMessage.value = '' }
}
const localizedMessage={success:(v:any)=>ElMessage.success(t(v)),error:(v:any)=>ElMessage.error(t(v)),warning:(v:any)=>ElMessage.warning(t(v))}
</script>

<style scoped>
.field-scope{color:#647982;font-size:13px;line-height:1.7;margin:0 0 20px}.real-data-mode :deep(.el-radio-group){display:flex;flex-direction:column;align-items:stretch;gap:6px}.real-data-mode :deep(.el-radio-button__inner){width:100%;border:1px solid #dce6e8!important;border-radius:5px!important}
.real-data-mode { padding: 4px 0; }
.real-data-mode .panel-card { max-width: 95%; margin-left: auto; margin-right: auto; }
.panel-card { border-radius: 8px; margin-bottom: 14px; }
.panel-card :deep(.el-card__header) { padding: 10px 16px; background-color: #fafafa; border-bottom: 1px solid #ebeef5; }
.panel-header { display: flex; align-items: center; gap: 6px; font-size: 14px; font-weight: 600; color: #303133; }
.panel-header .el-icon { color: #409eff; font-size: 16px; }
.upload-section { display: flex; flex-direction: column; gap: 12px; }
.upload-item { display: flex; flex-direction: column; gap: 4px; }
.upload-label { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 500; color: #303133; }
.upload-box { width: 100%; }
.upload-box :deep(.el-upload) { width: 100%; }
.upload-box :deep(.el-upload-dragger) { width: 100%; padding: 14px 8px; border-radius: 6px; }
.upload-icon { font-size: 24px; color: #c0c4cc; margin-bottom: 2px; }
.upload-text { font-size: 14px; color: #606266; }
.upload-text em { color: #409eff; font-style: normal; }
.upload-tip-wrap { display: flex; flex-direction: column; align-items: center; gap: 1px; }
.upload-tip-wrap code { font-size: 11px; white-space: nowrap; }
.chart-placeholder { height: 260px; background-color: #fafafa; border: 2px dashed #e4e7ed; border-radius: 8px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; color: #909399; font-size: 13px; transition: border-color .3s; }
.chart-placeholder:hover { border-color: #409eff; }
.result-card { margin-bottom: 0; }
.result-card :deep(.el-descriptions__label) { font-weight: 500; }
</style>

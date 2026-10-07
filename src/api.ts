/**
 * Axios 全局配置 — 统一后端 API 通信模块
 *
 * 使用方式：在其他组件中 import { apiClient, testConnection } from '@/api' 即可。
 */

import axios from 'axios'
import { ElMessage } from 'element-plus'

/** Axios 实例，默认指向 FastAPI 后端 */
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',   // 后端基础地址
  timeout: 30000,                      // 30 秒Timed out
  // 不全局强制 Content-Type：让 axios 按请求体自动推断
  //   - 纯对象 → application/json
  //   - FormData → multipart/form-data（自动带 boundary）
  //   - GET（无 body）→ 不携带 Content-Type，避免被浏览器判定为
  //     非简单请求而触发 CORS 预检，导致 /api/progress 轮询被拦截。
})

/** 全局拦截器：请求日志（调试用） */
apiClient.interceptors.request.use(
  (config) => {
    const fullUrl = `${config.baseURL ?? ''}${config.url ?? ''}`
    console.log(`[API] → ${config.method?.toUpperCase()} ${fullUrl}`)
    // 打印请求体：JSON 直接打印；FormData 遍历打印字段名与类型
    if (config.data != null) {
      if (typeof FormData !== 'undefined' && config.data instanceof FormData) {
        const parts: string[] = []
        config.data.forEach((value: any, key: string) => {
          if (value instanceof Blob) {
            parts.push(`${key}=<file:${value instanceof File ? value.name : 'blob'} ${value.size}B>`)
          } else {
            parts.push(`${key}=${value}`)
          }
        })
        console.log('[API]   data (FormData):', parts.join(', '))
      } else {
        console.log('[API]   data:', config.data)
      }
    }
    return config
  },
  (error) => Promise.reject(error),
)

/** 全局拦截器：响应日志（调试用） */
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API] ← ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    const url = error?.config?.url ?? '(unknown)'
    console.error(`[API] ✗ ${error.message} | URL: ${url}`)
    return Promise.reject(error)
  },
)

/**
 * 测试后端连接 — 页面加载时调用
 *
 * 向 GET / 发送请求，验证 FastAPI 是否正常启动。
 * 如果失败（后端未启动或端口不一致），弹出 ElMessage.error 提示用户。
 *
 * @returns 连接成功返回 true，失败返回 false
 */
export async function testConnection(): Promise<boolean> {
  try {
    const response = await apiClient.get('/')
    if (response.data?.status === 'ok') {
      console.log('[API] ✓ 后端连接成功', response.data)
      return true
    }
    console.warn('[API] ⚠ 后端响应异常', response.data)
    ElMessage.warning('Unexpected backend response')
    return false
  } catch (error: any) {
    const detail = error.message ?? 'Unknown error'
    console.error(`[API] ✗ 无法连接到后端服务器 (${detail})`)
    ElMessage.error(
      `Cannot reach ${apiClient.defaults.baseURL}. Start the local backend service.`
    )
    return false
  }
}

export default apiClient
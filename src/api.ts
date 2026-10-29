/** Shared backend client. Axios supplies JSON or multipart headers as needed. */
import axios from 'axios'
import { ElMessage } from 'element-plus'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
  timeout: 30000,
})

/** Check backend availability when the application mounts. */
export async function testConnection(): Promise<boolean> {
  try {
    const response = await apiClient.get('/')
    if (response.data?.status === 'ok') return true
    ElMessage.warning('Unexpected backend response')
    return false
  } catch {
    ElMessage.error(`Cannot reach ${apiClient.defaults.baseURL}. Start the local backend service.`)
    return false
  }
}

export default apiClient

import { ref, watch } from 'vue'
import translations from './translations.json'

type Language = 'en' | 'zh'
const preferenceKey = 'seaarray-language'
function savedLanguage(): Language {
  try { return localStorage.getItem(preferenceKey) === 'zh' ? 'zh' : 'en' } catch { return 'en' }
}
export const locale = ref<Language>(savedLanguage())
const zh: Record<string, string> = translations
const en: Record<string, string> = Object.fromEntries(Object.entries(zh).map(([a,b])=>[b,a]))
const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
// One pass keeps replacement text from being translated a second time.
const patterns = {
  zh: new RegExp('(?<![A-Za-z])(?:'+Object.keys(zh).sort((a,b)=>b.length-a.length).map(escape).join('|')+')(?![A-Za-z])','g'),
  en: new RegExp(Object.keys(en).sort((a,b)=>b.length-a.length).map(escape).join('|'),'g')
}
export function t<T>(value: T): T {
  if (typeof value !== 'string') return value
  const dict = locale.value === 'zh' ? zh : en
  return (dict[value] ?? value.replace(patterns[locale.value], key=>dict[key] ?? key)) as T
}
function localized(value: any): any {
  if (typeof value === 'string') return t(value)
  if (typeof value === 'function') return (...args: any[])=>t(value(...args))
  if (Array.isArray(value)) return value.map(localized)
  if (value && Object.getPrototypeOf(value) === Object.prototype)
    return Object.fromEntries(Object.entries(value).map(([key,val])=>[key,localized(val)]))
  return value
}
export function setLocalizedOption(chart: {setOption: (...args: any[])=>void}, options: any, settings?: any) {
  chart.setOption(localized(options),settings)
}
watch(locale, language=>{
  document.documentElement.lang=language === 'zh' ? 'zh-CN' : 'en'
  document.title=language === 'zh' ? 'SeaArray · 航迹设计' : 'SeaArray · Survey Design'
  try { localStorage.setItem(preferenceKey,language) } catch { /* Storage may be unavailable. */ }
},{immediate:true})

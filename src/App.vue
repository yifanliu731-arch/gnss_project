<!--
  App.vue - Seafloor Array Calibration 主界面
  左侧深蓝导航栏 + 顶部标题栏 + 中心淡蓝内容区
  淡蓝色海洋主题风格
-->
<template>
  <el-config-provider :locale="locale === 'zh' ? zhCn : en"><el-container class="app-shell">
    <!-- ==================== 左侧导航栏 ==================== -->
    <el-aside class="app-sidebar" width="220px">
      <!-- Logo 区域 -->
      <div class="sidebar-logo">
        <el-icon :size="28" color="#6cc4f5"><Ship /></el-icon>
        <span class="logo-text">{{ t("Array Calibration") }}</span>
        <span class="logo-sub">{{ t("SeaArray Calib") }}</span>
      </div>

      <!-- 导航菜单 -->
      <el-menu
        :default-active="activeTab"
        class="sidebar-menu"
        background-color="#0d3b66"
        text-color="#b8d4e8"
        active-text-color="#ffffff"
        @select="handleMenuSelect"
      >
        <el-menu-item index="simulation">
          <el-icon><Monitor /></el-icon>
          <span>{{ t("Survey design") }}</span>
          <span class="menu-sub">{{ t("Simulation") }}</span>
        </el-menu-item>

        <el-menu-item index="realdata">
          <el-icon><FolderOpened /></el-icon>
          <span>{{ t("Field data") }}</span>
          <span class="menu-sub">{{ t("Real Data") }}</span>
        </el-menu-item>
      </el-menu>

      <!-- 底部信息 -->
      <div class="sidebar-footer">
        <div class="footer-info">
          <el-icon :size="14"><InfoFilled /></el-icon>
          <span>{{ t("Joint seafloor positioning") }}</span>
        </div>
        <div class="footer-version">{{ t("RESEARCH WORKSPACE") }}</div>
      </div>
    </el-aside>

    <!-- ==================== 右侧主区域 ==================== -->
    <el-container class="app-main-area">
      <!-- 顶部标题栏 -->
      <el-header class="app-header">
        <div class="header-left">
          <el-icon :size="22"><Position /></el-icon>
          <h1 class="header-title">{{ t("Seafloor Array Calibration") }}</h1>
        </div>
        <div class="header-right"><el-radio-group v-model="locale" size="small" aria-label="Language / 语言"><el-radio-button value="zh">中文</el-radio-button><el-radio-button value="en">EN</el-radio-button></el-radio-group>
          <el-tag type="info" size="small" effect="plain">
            {{ t(activeTab === 'simulation' ? 'Survey design' : 'Field data') }}
          </el-tag>
        </div>
      </el-header>

      <!-- 主体内容区 -->
      <el-main class="app-content">
        <!-- 根据菜单切换显示对应组件 -->
        <KeepAlive><component :is="activeTab === 'simulation' ? SimulationMode : RealDataMode" /></KeepAlive>
      </el-main>
    </el-container>
  </el-container></el-config-provider>
</template>

<script setup lang="ts">
import { t, locale, setLocalizedOption } from "./i18n"
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import en from 'element-plus/es/locale/lang/en'
import { onMounted, ref } from 'vue'
import {
  Ship,
  Monitor,
  FolderOpened,
  InfoFilled,
  Position
} from '@element-plus/icons-vue'
import SimulationMode from './components/SimulationMode.vue'
import RealDataMode from './components/RealDataMode.vue'
import { testConnection } from './api'

/** 当前激活的导航项 */
const activeTab = ref<'simulation' | 'realdata'>('simulation')

/**
 * 菜单选择回调
 */
function handleMenuSelect(index: string) {
  activeTab.value = index as 'simulation' | 'realdata'
}

/** 页面加载时自动测试后端连接 */
onMounted(() => {
  testConnection()
})
</script>

<style scoped>
/* ================================================================ */
/*  整体 Shell                                                      */
/* ================================================================ */
.app-shell {
  height: 100vh;
  overflow: hidden;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ================================================================ */
/*  左侧导航栏                                                      */
/* ================================================================ */
.app-sidebar {
  background: linear-gradient(180deg, #0d3b66 0%, #0a2d4d 100%) !important;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: 2px 0 12px rgba(0, 40, 70, 0.15);
  z-index: 10;
}

/* --- Logo 区域 --- */
.sidebar-logo {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 24px 16px 18px;
  border-bottom: 1px solid rgba(108, 196, 245, 0.15);
}

.logo-text {
  font-size: 18px;
  font-weight: 700;
  color: #e0f0ff;
  letter-spacing: 2px;
  margin-top: 6px;
}

.logo-sub {
  font-size: 11px;
  color: rgba(184, 212, 232, 0.6);
  letter-spacing: 1px;
}

/* --- 导航菜单 --- */
.sidebar-menu {
  flex: 1;
  border-right: none;
  padding-top: 8px;
}

.sidebar-menu .el-menu-item {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  height: auto;
  padding: 14px 20px;
  margin: 2px 10px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
  transition: all 0.25s ease;
}

.sidebar-menu .el-menu-item:hover {
  background: rgba(20, 90, 140, 0.5) !important;
}

.sidebar-menu .el-menu-item.is-active {
  background: linear-gradient(135deg, #1a7ab5 0%, #2589d0 100%) !important;
  box-shadow: 0 2px 10px rgba(26, 122, 181, 0.4);
}

.sidebar-menu .menu-sub {
  font-size: 11px;
  color: rgba(184, 212, 232, 0.55);
  margin-left: 0;
  display: block;
  margin-top: 2px;
}

.sidebar-menu .el-icon {
  font-size: 18px;
  margin-bottom: 2px;
}

/* --- 底部信息 --- */
.sidebar-footer {
  padding: 16px 16px 20px;
  border-top: 1px solid rgba(108, 196, 245, 0.12);
}

.footer-info {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: rgba(184, 212, 232, 0.55);
}

.footer-version {
  font-size: 10px;
  color: rgba(184, 212, 232, 0.3);
  margin-top: 4px;
  padding-left: 20px;
}

/* ================================================================ */
/*  右侧主区域                                                      */
/* ================================================================ */
.app-main-area {
  background: #e8f4fd;
  flex-direction: column;
}

/* --- 顶部标题栏 --- */
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, #0a2d4d 0%, #0d3b66 100%);
  padding: 0 28px;
  height: 56px;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0, 30, 60, 0.2);
  z-index: 5;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #e0f0ff;
}

.header-title {
  font-size: 18px;
  font-weight: 600;
  letter-spacing: 3px;
  color: #e0f0ff;
  margin: 0;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* --- 主体内容区 --- */
.app-content {
  flex: 1;
  padding: 16px 20px;
  overflow-y: auto;
  background: #e8f4fd;
}

/* ================================================================ */
/*  响应式：小屏收起侧边栏                                          */
/* ================================================================ */
@media (max-width: 768px) {
  .app-sidebar {
    width: 60px !important;
    min-width: 60px !important;
  }

  .logo-text,
  .logo-sub,
  .menu-sub,
  .sidebar-footer,
  .sidebar-menu span {
    display: none;
  }

  .sidebar-menu .el-menu-item {
    align-items: center;
    padding: 14px 10px;
    margin: 2px 4px;
  }

  .sidebar-logo {
    padding: 14px 4px;
  }

  .header-title {
    font-size: 14px;
    letter-spacing: 1px;
  }

  .app-content {
    padding: 12px;
  }
}

.app-sidebar{background:#173943!important;box-shadow:none;width:184px!important}
.sidebar-logo{align-items:flex-start;padding:28px 22px;gap:7px}.logo-text{font-size:16px;letter-spacing:1px}.logo-sub{letter-spacing:1.6px}
.sidebar-menu{background:transparent!important}.sidebar-menu .el-menu-item{font-size:13px;padding:15px 14px!important;margin:5px 12px}
.sidebar-menu .el-menu-item.is-active{background:#2b555d!important;box-shadow:none}.sidebar-menu .el-menu-item:hover{background:#234b55!important}
.app-header{background:#fff;border-bottom:1px solid #dce6e8;box-shadow:none;height:64px;padding:0 30px}.header-title{color:#345963;font-size:15px;letter-spacing:1px}.header-left{color:#4b7b81}
.app-main-area,.app-content{background:#f4f7f8}.app-content{padding:28px 32px}.sidebar-footer{font-size:11px}.footer-version{letter-spacing:1px;font-size:9px}
@media(max-width:768px){.app-sidebar{width:60px!important}.sidebar-logo{padding:20px 14px}.sidebar-menu .el-menu-item{margin:4px;padding:14px 10px!important}.app-content{padding:18px 12px}.app-header{padding:0 16px}.header-title{letter-spacing:0;font-size:13px}}

@media(max-width:768px){.header-right{gap:6px}.header-right>.el-tag{display:none}.header-title{font-size:12px}}
</style>
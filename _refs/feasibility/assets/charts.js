// assets/charts.js
(function() {
  'use strict';

  // --- Read CSS variables ---
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim() || '#1a56c4';
  var accent2 = style.getPropertyValue('--accent2').trim() || '#c0392b';
  var ink = style.getPropertyValue('--ink').trim() || '#1a1a2e';
  var muted = style.getPropertyValue('--muted').trim() || '#6b7280';
  var rule = style.getPropertyValue('--rule').trim() || '#e2e5e9';
  var bg2 = style.getPropertyValue('--bg2').trim() || '#ffffff';

  // --- Initialize Mermaid ---
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'base',
      themeVariables: {
        primaryColor: '#eef4ff',
        primaryTextColor: ink,
        primaryBorderColor: accent,
        lineColor: accent,
        secondaryColor: '#fef9e7',
        tertiaryColor: '#eafaf1',
        fontFamily: '"Noto Sans CJK SC", "WenQuanYi Micro Hei", sans-serif',
        fontSize: '14px'
      },
      securityLevel: 'loose',
      flowchart: {
        htmlLabels: true,
        curve: 'basis',
        rankSpacing: 40,
        nodeSpacing: 30
      }
    });
  }

  // --- Chart: CPFE Maturity Radar ---
  var maturityEl = document.getElementById('chart-maturity');
  if (maturityEl && typeof echarts !== 'undefined') {
    var chart = echarts.init(maturityEl, null, { renderer: 'svg' });

    var indicators = [
      { name: '屈服强度预测', max: 10 },
      { name: '抗拉强度预测', max: 10 },
      { name: '疲劳寿命预测', max: 10 },
      { name: 'AM微观组织纳入', max: 10 },
      { name: '残余应力预测', max: 10 },
      { name: '多道多层扩展', max: 10 },
      { name: '计算效率', max: 10 },
      { name: '磨损预测', max: 10 }
    ];

    // 基于文献调研的成熟度评分（1-10分）
    // 数据来源：Nomoto等(2023), Wang等(2024), Huang等(2024), MPF-CPFEM, Cui等(2024)
    var data = [
      {
        value: [7, 7, 3, 6, 5, 2, 2, 1],
        name: '当前成熟度',
        itemStyle: { color: accent },
        areaStyle: { opacity: 0.15 },
        lineStyle: { width: 2 }
      },
      {
        value: [9, 9, 7, 8, 8, 7, 8, 6],
        name: '工程应用目标',
        itemStyle: { color: accent2, type: 'dashed' },
        areaStyle: { opacity: 0.05 },
        lineStyle: { width: 1.5, type: 'dashed' }
      }
    ];

    chart.setOption({
      animation: false,
      tooltip: {
        trigger: 'item',
        appendToBody: true,
        formatter: function(params) {
          var html = '<strong>' + params.name + '</strong><br/>';
          var vals = params.value;
          for (var i = 0; i < indicators.length; i++) {
            html += indicators[i].name + ': ' + vals[i] + '/10<br/>';
          }
          return html;
        }
      },
      legend: {
        data: ['当前成熟度', '工程应用目标'],
        bottom: 10,
        textStyle: { color: muted, fontSize: 12 },
        itemGap: 30
      },
      radar: {
        indicator: indicators,
        center: ['50%', '48%'],
        radius: '65%',
        splitNumber: 5,
        axisName: {
          color: ink,
          fontSize: 12,
          fontWeight: 600,
          padding: [3, 5]
        },
        splitLine: {
          lineStyle: { color: rule, width: 1 }
        },
        splitArea: {
          areaStyle: {
            color: ['rgba(26,86,196,0.02)', 'rgba(26,86,196,0.04)', 'rgba(26,86,196,0.06)', 'rgba(26,86,196,0.08)', 'rgba(26,86,196,0.10)']
          }
        },
        axisLine: {
          lineStyle: { color: rule }
        }
      },
      series: [{
        type: 'radar',
        data: data,
        symbolSize: 5
      }]
    });

    window.addEventListener('resize', function() { chart.resize(); });
  }

})();

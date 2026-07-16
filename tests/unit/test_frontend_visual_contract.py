from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
INDEX = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
STYLE = (ROOT / "web" / "static" / "style.css").read_text(encoding="utf-8")
SCRIPT = (ROOT / "web" / "static" / "app.js").read_text(encoding="utf-8")


def css_blocks(selector: str) -> list[str]:
    return re.findall(rf"{re.escape(selector)}\s*\{{([^}}]+)\}}", STYLE, re.DOTALL)


def assert_rule(selector: str, *declarations: str) -> None:
    blocks = css_blocks(selector)
    assert blocks, f"缺少 CSS 规则: {selector}"
    assert any(all(item in block for item in declarations) for block in blocks), (
        f"{selector} 未在同一规则中包含: {declarations}"
    )


def test_critical_business_dom_ids_remain_unique():
    ids = re.findall(r'\bid="([^"]+)"', INDEX)
    assert len(ids) == len(set(ids))
    required = {
        "browseBtn",
        "startBtn",
        "btBrowseStrategyBtn",
        "btStartBtn",
        "btEquityChart",
        "btRollingChart",
        "rtSourceSelect",
        "rtTimeframeSelect",
        "rtStrategySelect",
        "rtBrowseStrategyBtn",
        "rtAddBtn",
        "rtGrid",
        "rtFeishuSaveBtn",
    }
    assert required <= set(ids)


def test_three_pages_share_titles_sidebar_and_file_action_component():
    assert INDEX.count('class="page-header"') == 3
    for title in ("01 模型训练", "02 策略回测", "03 实时分析"):
        assert title in INDEX
    assert INDEX.count("file-action-button") == 3
    assert "选择数据文件" in INDEX
    assert "选择策略 JSON" in INDEX
    assert "导入策略 JSON" in INDEX


def test_backtest_tabs_are_real_accessible_panels_with_exact_names():
    expected = {
        "btRollingTab": ("btRollingPanel", "滚动夏普"),
        "btTradesTab": ("btTablePanel", "交易明细"),
        "btLogsTab": ("btLogPanel", "日志"),
        "btExplainTab": ("btHowtoPanel", "通俗解释"),
    }
    for tab_id, (panel_id, label) in expected.items():
        button = re.search(
            rf'<button[^>]+id="{tab_id}"[^>]+aria-controls="{panel_id}"[^>]*>{label}</button>',
            INDEX,
        )
        assert button, f"标签 {label} 未正确绑定内容面板"
        assert re.search(
            rf'<section[^>]+id="{panel_id}"[^>]+role="tabpanel"[^>]+data-bt-tab-panel',
            INDEX,
        )
    assert "滚动盈亏" not in INDEX
    assert "data-bt-jump" not in INDEX
    assert_rule(".bt-result-tab", "flex: 0 0 auto", "width: auto", "font-size: 17px")
    assert_rule(".bt-result-tabs", "height: 52px", "margin: 0", "border-bottom: 1px solid #808488")


def test_backtest_tabs_switch_panels_in_place_and_support_keyboard():
    assert 'const panelId = btn.getAttribute("aria-controls")' in SCRIPT
    assert 'panel.hidden = panel.id !== panelId' in SCRIPT
    assert 'event.key === "ArrowRight"' in SCRIPT
    assert 'event.key === "ArrowLeft"' in SCRIPT
    assert 'event.key === "Home"' in SCRIPT
    assert 'event.key === "End"' in SCRIPT
    assert_rule(".bt-tabs-panel", "padding: 0", "min-height: 480px")
    assert_rule(".bt-tab-stage", "min-height: 426px", "background: #0f1418")
    assert SCRIPT.index("bindWorkspaceNavigation();") < SCRIPT.index("await ensureControlToken();")


def test_equity_title_connects_directly_to_deep_chart_surface():
    assert_rule(".bt-charts-panel", "min-height: 480px", "padding: 0")
    assert_rule(".bt-charts-panel > .panel-head", "margin: 0")
    assert_rule(".bt-charts-panel > .equity-live", "padding: 24px", "background: #0f1418")
    assert_rule("#btEquityEmpty", "min-height: 368px", "margin: 0", "background: #0f1418")


def test_large_spacious_visual_rules_are_selector_specific():
    assert_rule(".header", "width: 288px", "height: 98px", "min-height: 98px")
    assert_rule(".brand h1", "font-size: 28px", "line-height: 36px", "font-weight: 700")
    assert_rule(".page-header h1", "font-size: 26px", "line-height: 36px", "font-weight: 700")
    assert_rule(".file-action-button", "height: 52px", "font-size: 17px", "border: 1px solid #e5e7eb")
    assert_rule(".notice-bar", "min-height: 52px", "font-size: 17px", "border: 1px solid #e5e7eb")
    assert_rule(".panel", "padding: 24px", "margin-bottom: 24px", "border: 1px solid #808488")


def test_mobile_header_can_expand_without_horizontal_compression():
    mobile = STYLE[STYLE.index("@media (max-width: 860px)") :]
    header = re.search(r"\.header\s*\{([^}]+)\}", mobile, re.DOTALL)
    assert header
    for declaration in ("height: auto", "min-height: 98px", "flex-wrap: wrap"):
        assert declaration in header.group(1)
    assert re.search(r"\.header-meta\s*\{[^}]*flex-wrap: wrap", mobile, re.DOTALL)
    compact = STYLE[STYLE.index("@media (max-width: 640px)") :]
    assert "body { font-size: 16px" not in compact
    assert ".bt-result-tab { padding-inline: 22px" not in compact


def test_realtime_placeholder_and_missing_value_guards_exist():
    for placeholder in ("请选择数据源", "请选择周期", "请选择策略"):
        assert placeholder in SCRIPT
    assert 'if (!source)' in SCRIPT
    assert 'if (!timeframe)' in SCRIPT
    assert 'syncSelectPlaceholder' in SCRIPT
    assert 'function setRtStrategyHint' in SCRIPT
    assert 'picked.classList.toggle("bad", isError)' in SCRIPT
    assert '$("rtSymbolInput").addEventListener("input", onRtStrategyChange)' in SCRIPT


def test_sidebar_resource_targets_exist_and_are_wired():
    targets = re.findall(r'data-resource-target="([^"]+)"', INDEX)
    assert targets == ["trainingLogPanel", "strategiesPanel", "howtoPanel", "aiPanelSection"]
    for target in targets:
        assert f'id="{target}"' in INDEX
    assert 'document.querySelectorAll(".sidebar-resource")' in SCRIPT

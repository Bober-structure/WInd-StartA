import io
import math
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==========================================
# 1. СПРАВОЧНИКИ СП 20.13330.2016 (Изм. 2, 5)
# ==========================================
WIND_REGIONS = {
    "Ia": 0.18, "I": 0.23, "II": 0.30, "III": 0.38,
    "IV": 0.48, "V": 0.60, "VI": 0.73, "VII": 0.85
}

TABLE_11_3 = {
    'A': {'k10': 1.00, 'alpha': 0.15, 'zeta10': 0.76, 'z_min': 10.0, 'desc': 'Открытые побережья морей, озёр, степи'},
    'B': {'k10': 0.65, 'alpha': 0.20, 'zeta10': 1.06, 'z_min': 10.0, 'desc': 'Городские территории, лесные массивы'},
    'C': {'k10': 0.40, 'alpha': 0.25, 'zeta10': 1.78, 'z_min': 10.0, 'desc': 'Городские районы с застройкой > 25 м'}
}

DEFAULT_ZONES = [
    {"Зона": "I", "Сторона": "d (по ширине)", "c": -1.2},
    {"Зона": "II", "Сторона": "d (по ширине)", "c": -0.8},
    {"Зона": "III", "Сторона": "d (по ширине)", "c": -0.5},
    {"Зона": "IV", "Сторона": "d (по ширине)", "c": 0.8},
    {"Зона": "V", "Сторона": "d (по ширине)", "c": -0.5},
    {"Зона": "VI", "Сторона": "l (по длине)", "c": -1.8},
    {"Зона": "VII", "Сторона": "l (по длине)", "c": -1.2},
    {"Зона": "VIII", "Сторона": "l (по длине)", "c": -0.7},
    {"Зона": "IX", "Сторона": "l (по длине)", "c": -0.5},
    {"Зона": "X", "Сторона": "l (по длине)", "c": -0.6},
    {"Зона": "XI", "Сторона": "l (по длине)", "c": -1.0},
    {"Зона": "XII", "Сторона": "l (по длине)", "c": -0.5},
]

DEFAULT_LEVELS = [
    0.0, 4.05, 8.25, 11.7, 16.2, 25.05, 29.25, 33.45, 37.65, 41.85, 46.05, 50.25, 
    54.45, 58.65, 62.85, 67.05, 71.25, 75.45, 79.65, 83.85, 88.05, 92.25, 96.45, 
    100.65, 104.85, 109.05, 113.25, 117.05, 125.25, 130.05, 138.45, 142.65, 146.85, 
    151.05, 155.25, 159.45, 163.65, 167.85, 172.05, 176.25, 180.05, 184.65, 188.85, 196.75
]

# ==========================================
# 2. РАСЧЁТНЫЕ ФУНКЦИИ И ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ
# ==========================================
def fmt_num(val: float, decimals: int = 3) -> str:
    """Форматирует число с запятой."""
    return f"{val:.{decimals}f}".replace('.', ',')

def get_k_z(z: float, terrain: str) -> float:
    params = TABLE_11_3[terrain]
    k10 = params['k10']
    alpha = params['alpha']
    z_eff = max(z, params['z_min'])
    return k10 * ((z_eff / 10.0) ** (2 * alpha))

def get_ze_for_height(z: float, h: float, b_dim: float) -> tuple[float, str]:
    """Расчёт эквивалентной высоты z_e по п. 11.1.5 СП 20"""
    if h <= b_dim:
        return h, f"h ≤ B (вся высота z_e = h)"
    elif b_dim < h <= 2 * b_dim:
        if z <= h - b_dim:
            return h - b_dim, f"B < h ≤ 2B (нижняя зона: z ≤ {h-b_dim:.2f}м)"
        else:
            return h, f"B < h ≤ 2B (верхняя зона: z > {h-b_dim:.2f}м)"
    else:
        if z <= b_dim:
            return b_dim, f"h > 2B (нижняя зона: z ≤ {b_dim:.2f}м)"
        elif z >= h - b_dim:
            return h, f"h > 2B (верхняя зона: z ≥ {h-b_dim:.2f}м)"
        else:
            return z, f"h > 2B (средняя зона: {b_dim:.2f}м < z < {h-b_dim:.2f}м)"

def get_zone_dim(side_str: str, d_val: float, l_val: float) -> tuple[float, str, str]:
    """Возвращает тупл: (значение размера, текстовое описание, символьный код)"""
    s = str(side_str).lower()
    if "l" in s or "длин" in s:
        return l_val, f"l = {l_val:.2f} м (по длине здания)", "l"
    return d_val, f"d = {d_val:.2f} м (по ширине здания)", "d"

def calculate_h_grp(all_raw_levels: list[float]) -> dict[float, float]:
    sorted_all = sorted(list(set([float(v) for v in all_raw_levels if float(v) >= 0])))
    if not sorted_all:
        return {}
    positive_levels = [z for z in sorted_all if z > 0]
    h_grp_dict = {}
    n = len(sorted_all)
    for z in positive_levels:
        idx = sorted_all.index(z)
        if idx == n - 1:
            prev_z = sorted_all[idx - 1] if idx > 0 else 0.0
            h_grp = z - prev_z
        else:
            next_z = sorted_all[idx + 1]
            prev_z = sorted_all[idx - 1] if idx > 0 else 0.0
            h_grp = ((next_z - z) / 2.0) + ((z - prev_z) / 2.0)
        h_grp_dict[z] = h_grp
    return h_grp_dict

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Конвертирует DataFrame в CSV с разделителем ';' и запятой в числах."""
    export_df = df.copy()
    for col in export_df.columns:
        export_df[col] = export_df[col].astype(str).str.replace('.', ',', regex=False)
    return export_df.to_csv(index=False, sep=';').encode('utf-8-sig')

def render_custom_table(df: pd.DataFrame, subheaders: dict = None):
    """Вывод HTML-таблицы с фиксированными стилями"""
    html = """
    <style>
        .table-container {
            max-height: 550px;
            overflow-y: auto;
            overflow-x: auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            margin: 10px 0;
        }
        .custom-excel-table {
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 14px;
        }
        .custom-excel-table thead {
            position: sticky;
            top: 0;
            z-index: 2;
        }
        .custom-excel-table th {
            background-color: rgb(31, 78, 120) !important;
            color: #ffffff !important;
            font-weight: bold;
            padding: 8px 12px;
            text-align: center;
            border: 1px solid #d0d0d0;
        }
        .custom-excel-table th.sub-header {
            background-color: #285078 !important;
            color: #dce7f5 !important;
            font-weight: 500;
            font-size: 12px;
            padding: 4px 8px;
            border: 1px solid #3c6590;
        }
        .custom-excel-table td {
            padding: 6px 12px;
            border: 1px solid #e0e0e0;
            text-align: center;
        }
        .custom-excel-table tr:nth-child(even) {
            background-color: #f9fbfd;
        }
        .custom-excel-table tr:hover {
            background-color: #f1f5f9;
        }
    </style>
    <div class="table-container">
    <table class="custom-excel-table">
        <thead>
            <tr>
    """
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"

    if subheaders:
        html += "<tr>"
        for col in df.columns:
            sub_txt = subheaders.get(col, "—")
            html += f"<th class='sub-header'>{sub_txt}</th>"
        html += "</tr>"

    html += "</thead><tbody>"
    
    for _, row in df.iterrows():
        html += "<tr>"
        for val in row:
            html += f"<td>{val}</td>"
        html += "</tr>"
        
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ
# ==========================================
st.set_page_config(page_title="Калькулятор ветровых нагрузок СП 20", page_icon="🌪️", layout="wide")
st.title("🌪️ Калькулятор ветровой нагрузки по СП 20.13330.2016 (с Изм. 2, 5)")

if "df_levels" not in st.session_state:
    st.session_state.df_levels = pd.DataFrame({"Отметка z (м)": DEFAULT_LEVELS})

if "df_zones" not in st.session_state:
    st.session_state.df_zones = pd.DataFrame(DEFAULT_ZONES)

if "calculated" not in st.session_state:
    st.session_state.calculated = False

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Результаты и пошаговый расчёт", 
    "🏢 Настройка уровней (z)", 
    "🧱 Настройка зон", 
    "📈 Эпюры и графики",
    "📐 Нагрузка с учётом грузовой площади (Wp)"
])

# ------------------------------------------
# ВКЛАДКА 2: НАСТРОЙКА УРОВНЕЙ Z
# ------------------------------------------
with tab2:
    st.subheader("Настройка высотных отметок этажей / уровней z (м)")
    st.info("💡 Вы можете вставлять скопированные данные из Excel прямо в таблицу через **Ctrl+V**, либо использовать форму импорта ниже.")

    col_btn1, _ = st.columns([1, 3])
    if col_btn1.button("🔄 Сбросить отметки к 44 по умолчанию"):
        st.session_state.df_levels = pd.DataFrame({"Отметка z (м)": DEFAULT_LEVELS})
        if "levels_editor" in st.session_state:
            del st.session_state["levels_editor"]
        st.session_state.calculated = False
        st.rerun()

    with st.expander("📋 Быстрый импорт отметок (Загрузка файла Excel/CSV или вставка текста)"):
        up_file_lvl = st.file_uploader("Загрузить файл Excel (.xlsx) или CSV с отметками", type=["xlsx", "csv"], key="file_lvl")
        if up_file_lvl is not None:
            try:
                df_imp = pd.read_csv(up_file_lvl) if up_file_lvl.name.endswith(".csv") else pd.read_excel(up_file_lvl)
                if not df_imp.empty:
                    col0 = df_imp.columns[0]
                    vals = pd.to_numeric(df_imp[col0].astype(str).str.replace(',', '.'), errors='coerce').dropna().tolist()
                    st.session_state.df_levels = pd.DataFrame({"Отметка z (м)": vals})
                    if "levels_editor" in st.session_state:
                        del st.session_state["levels_editor"]
                    st.success(f"Загружено {len(vals)} отметок из файла!")
                    st.rerun()
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")

        raw_txt_lvl = st.text_area("Или вставьте скопированный из Excel столбец отметок:", placeholder="0,0\n4,05\n8,25\n11,7", height=120)
        if st.button("Применить вставленный текст (Отметки)"):
            if raw_txt_lvl.strip():
                try:
                    lines = [float(x.replace(',', '.')) for x in raw_txt_lvl.strip().split() if x.strip()]
                    st.session_state.df_levels = pd.DataFrame({"Отметка z (м)": lines})
                    if "levels_editor" in st.session_state:
                        del st.session_state["levels_editor"]
                    st.success(f"Загружено {len(lines)} отметок!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка формата чисел: {e}")

    with st.form(key="levels_form"):
        edited_levels_df = st.data_editor(
            st.session_state.df_levels,
            num_rows="dynamic",
            column_config={
                "Отметка z (м)": st.column_config.NumberColumn(
                    "Отметка z (м)", format="%.3f", min_value=0.0, max_value=500.0, step=0.001, alignment="left"
                )
            },
            use_container_width=True,
            key="levels_editor"
        )
        submit_levels = st.form_submit_button("▶️ Сохранить отметки и выполнить расчёт", type="primary", use_container_width=True)

# ------------------------------------------
# ВКЛАДКА 3: НАСТРОЙКА АЭРОДИНАМИЧЕСКИХ ЗОН
# ------------------------------------------
with tab3:
    st.subheader("🧱 Настройка аэродинамических зон")
    st.info("💡 Укажите принадлежность каждой зоны к стороне здания ($d$ или $l$) и коэффициент $c$.")

    col_z_left, col_z_right = st.columns([3, 2])

    with col_z_left:
        col_btn_z, _ = st.columns([1, 1])
        if col_btn_z.button("🔄 Сбросить зоны к умолчанию"):
            st.session_state.df_zones = pd.DataFrame(DEFAULT_ZONES)
            if "zones_editor" in st.session_state:
                del st.session_state["zones_editor"]
            st.session_state.calculated = False
            st.rerun()

        with st.form(key="zones_form"):
            edited_zones_df = st.data_editor(
                st.session_state.df_zones,
                num_rows="dynamic",
                column_config={
                    "Зона": st.column_config.TextColumn("Зона", alignment="center", width="small"),
                    "Сторона": st.column_config.SelectboxColumn(
                        "Сторона здания",
                        options=["d (по ширине)", "l (по длине)"],
                        default="d (по ширине)",
                        required=True,
                        width="medium"
                    ),
                    "c": st.column_config.NumberColumn(
                        "Коэффициент c", format="%.2f", min_value=-3.0, max_value=3.0, step=0.05, alignment="center", width="small"
                    )
                },
                use_container_width=True,
                key="zones_editor"
            )
            submit_zones = st.form_submit_button("▶️ Сохранить зоны и выполнить расчёт", type="primary", use_container_width=True)

    with col_z_right:
        with st.expander("📋 Быстрый импорт зон (Файл / Текст)", expanded=True):
            st.caption("Формат столбцов: Зона | Сторона (d/l) | Коэффициент c")
            up_file_zn = st.file_uploader("Загрузить Excel (.xlsx) или CSV", type=["xlsx", "csv"], key="file_zn")
            if up_file_zn is not None:
                try:
                    df_imp_z = pd.read_csv(up_file_zn) if up_file_zn.name.endswith(".csv") else pd.read_excel(up_file_zn)
                    if not df_imp_z.empty:
                        st.session_state.df_zones = df_imp_z
                        if "zones_editor" in st.session_state:
                            del st.session_state["zones_editor"]
                        st.success("Зоны успешно загружены из файла!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Ошибка чтения файла: {e}")

            raw_txt_zn = st.text_area(
                "Или вставьте скопированную из Excel таблицу:",
                placeholder="I\td\t-1,2\nII\td\t-0,8\nVI\tl\t-1,8",
                height=130
            )
            if st.button("Применить вставленный текст (Зоны)"):
                if raw_txt_zn.strip():
                    try:
                        rows = []
                        for line in raw_txt_zn.strip().split('\n'):
                            parts = [p.strip() for p in line.split('\t')]
                            if len(parts) >= 3:
                                side_val = "l (по длине)" if ("l" in parts[1].lower() or "длин" in parts[1].lower()) else "d (по ширине)"
                                rows.append({"Зона": parts[0], "Сторона": side_val, "c": float(parts[2].replace(',', '.'))})
                            elif len(parts) == 2:
                                side_val = "l (по длине)" if ("l" in parts[1].lower() or "длин" in parts[1].lower()) else "d (по ширине)"
                                rows.append({"Зона": parts[0], "Сторона": side_val, "c": 0.0})
                            elif len(parts) == 1 and parts[0]:
                                rows.append({"Зона": f"Зона {len(rows)+1}", "Сторона": "d (по ширине)", "c": float(parts[0].replace(',', '.'))})
                        if rows:
                            st.session_state.df_zones = pd.DataFrame(rows)
                            if "zones_editor" in st.session_state:
                                del st.session_state["zones_editor"]
                            st.success(f"Загружено {len(rows)} зон!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка разбора текста: {e}")

# ------------------------------------------
# БОКОВОЕ МЕНЮ
# ------------------------------------------
with st.sidebar.form(key="input_form"):
    st.header("⚙️ Исходные параметры")
    
    wind_region_keys = list(WIND_REGIONS.keys())
    region = st.selectbox("Ветровой район", wind_region_keys, index=wind_region_keys.index("I"))
    w0 = WIND_REGIONS[region]

    terrain = st.radio("Тип местности (Табл. 11.3)", ["A", "B", "C"], index=1)

    st.subheader("📐 Габариты здания")
    d = st.number_input("Ширина здания d (поперек ветра, м)", min_value=1.0, max_value=500.0, value=48.0, step=0.5)
    l = st.number_input("Длина здания l (вдоль ветра, м)", min_value=1.0, max_value=500.0, value=61.2, step=0.1)
    gamma_f = st.number_input("Коэффициент надежности γf", min_value=1.0, max_value=2.0, value=1.4, step=0.05)

    submit_button = st.form_submit_button(label="▶️ Выполнить расчёт", type="primary", use_container_width=True)

# ------------------------------------------
# ФИКСАЦИЯ ИЗМЕНЕНИЙ И ОБНОВЛЕНИЕ СОСТОЯНИЯ
# ------------------------------------------
if submit_button or submit_levels or submit_zones:
    st.session_state.df_levels = edited_levels_df
    st.session_state.df_zones = edited_zones_df
    st.session_state.calculated = True

# Подготовка расчётных списков
raw_levels_list = sorted([float(val) for val in st.session_state.df_levels["Отметка z (м)"].dropna().tolist() if float(val) >= 0])
levels_list = [z for z in raw_levels_list if z > 0]
h_max = max(raw_levels_list) if raw_levels_list else 10.0

st.sidebar.caption(f"Давление $w_0 = {w0}$ кПа")
st.sidebar.caption(f"Описание местности: {TABLE_11_3[terrain]['desc']}")
st.sidebar.metric(label="Высота здания h = max(z)", value=fmt_num(h_max, 3) + " м")

# ------------------------------------------
# ВКЛАДКА 1: Результаты и пошаговый расчёт
# ------------------------------------------
with tab1:
    if not st.session_state.calculated:
        st.info("💡 Настройте необходимые параметры в левом меню или на вкладках и нажмите **«▶️ Выполнить расчёт»**.")
    else:
        # Расчёт основных данных для таблицы Wd
        table_data = []
        subheaders_t1 = {
            "№": "—",
            "Отметка z (м)": "—",
            "z_e(d) (м)": f"d = {fmt_num(d, 2)} м",
            "z_e(l) (м)": f"l = {fmt_num(l, 2)} м",
        }

        for _, z_row in st.session_state.df_zones.iterrows():
            zone_name = str(z_row['Зона']) if pd.notna(z_row['Зона']) else ""
            side_str = str(z_row.get('Сторона', 'd')) if pd.notna(z_row.get('Сторона')) else 'd'
            dim_val, _, dim_code = get_zone_dim(side_str, d, l)
            subheaders_t1[f"Зона {zone_name} (кПа)"] = f"{dim_code} = {fmt_num(dim_val, 2)} м"

        for idx, z_val in enumerate(levels_list, 1):
            ze_d, _ = get_ze_for_height(z_val, h_max, d)
            ze_l, _ = get_ze_for_height(z_val, h_max, l)
            
            row = {
                "№": int(idx),
                "Отметка z (м)": fmt_num(z_val, 3),
                "z_e(d) (м)": fmt_num(ze_d, 3),
                "z_e(l) (м)": fmt_num(ze_l, 3),
            }
            for _, z_row in st.session_state.df_zones.iterrows():
                zone_name = str(z_row['Зона']) if pd.notna(z_row['Зона']) else ""
                side_str = str(z_row.get('Сторона', 'd')) if pd.notna(z_row.get('Сторона')) else 'd'
                c_val = float(z_row['c']) if pd.notna(z_row['c']) else 0.0
                
                dim_val, _, _ = get_zone_dim(side_str, d, l)
                ze_zone, _ = get_ze_for_height(z_val, h_max, dim_val)
                k_ze = get_k_z(ze_zone, terrain)
                
                wd = w0 * k_ze * c_val * gamma_f
                row[f"Зона {zone_name} (кПа)"] = fmt_num(wd, 3)
            table_data.append(row)
            
        df_result = pd.DataFrame(table_data)

        # Заголовок и кнопка экспорта справа
        col_h1, col_exp1 = st.columns([3, 1])
        with col_h1:
            st.subheader(f"1. Сводная таблица расчётных ветровых нагрузок $w_d$ (кПа) [h = {fmt_num(h_max, 3)} м, d = {fmt_num(d, 2)} м, l = {fmt_num(l, 2)} м]")
        with col_exp1:
            csv_bytes_t1 = convert_df_to_csv(df_result)
            st.download_button(
                label="📥 Скачать в CSV (.csv)",
                data=csv_bytes_t1,
                file_name="ветровая_нагрузка_wd.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download_t1"
            )

        st.caption("Примечание: В подзаголовках столбцов указан учтённый размер стороны здания для каждой зоны.")
        render_custom_table(df_result, subheaders_t1)

        st.markdown("---")
        st.subheader("2. 🔍 Подробный пошаговый расчёт для выбранной отметки")
        
        if levels_list and not st.session_state.df_zones.empty:
            col_sel1, col_sel2 = st.columns(2)
            selected_z_idx = col_sel1.selectbox(
                "Выберите уровень / отметку z:", 
                range(len(levels_list)), 
                format_func=lambda i: f"№ {i+1}: z = {fmt_num(levels_list[i], 3)} м"
            )
            
            zone_options = [str(z) for z in st.session_state.df_zones['Зона'].dropna().tolist()]
            selected_zone_name = col_sel2.selectbox(
                "Выберите аэродинамическую зону:", 
                zone_options
            )
            
            z_sel = levels_list[selected_z_idx]
            sel_zone_row = st.session_state.df_zones[st.session_state.df_zones['Зона'].astype(str) == selected_zone_name].iloc[0]
            c_val = float(sel_zone_row['c']) if pd.notna(sel_zone_row['c']) else 0.0
            side_str = str(sel_zone_row.get('Сторона', 'd'))
            
            dim_val, dim_label, dim_code = get_zone_dim(side_str, d, l)
            ze_sel, zone_desc_sel = get_ze_for_height(z_sel, h_max, dim_val)
            
            t_params = TABLE_11_3[terrain]
            k10 = t_params['k10']
            alpha = t_params['alpha']
            two_alpha = 2 * alpha
            
            ze_eff = max(ze_sel, t_params['z_min'])
            k_ze_val = get_k_z(ze_sel, terrain)
            wm_val = w0 * k_ze_val * c_val
            wd_val = wm_val * gamma_f
            
            st.info(
                f"**Отметка z = {fmt_num(z_sel, 3)} м** | "
                f"**Зона {selected_zone_name}** ($c = {c_val}$) | "
                f"**Сторона здания**: {dim_label} | "
                f"**Эквивалентная высота $z_e = {fmt_num(ze_sel, 3)}$ м** ({zone_desc_sel})"
            )
            
            if h_max > 2 * dim_val:
                h_cond_str = f"h = {fmt_num(h_max, 3)} \\text{{ м}} > 2 \\cdot {dim_code} = {fmt_num(2*dim_val, 2)} \\text{{ м}}"
                if z_sel <= dim_val:
                    ze_expr = f"z_e = {dim_code} = {fmt_num(dim_val, 2)} \\text{{ м}}"
                    ze_zone_info = f"нижняя зона: $z \\le {fmt_num(dim_val, 2)}$ м"
                elif z_sel >= h_max - dim_val:
                    ze_expr = f"z_e = h = {fmt_num(h_max, 3)} \\text{{ м}}"
                    ze_zone_info = f"верхняя зона: $z \\ge {fmt_num(h_max-dim_val, 2)}$ м"
                else:
                    ze_expr = f"z_e = z = {fmt_num(z_sel, 3)} \\text{{ м}}"
                    ze_zone_info = f"средняя зона: ${fmt_num(dim_val, 2)} \\text{{ м}} < z < {fmt_num(h_max-dim_val, 2)} \\text{{ м}}$"
            elif h_max <= dim_val:
                h_cond_str = f"h = {fmt_num(h_max, 3)} \\text{{ м}} \\le {dim_code} = {fmt_num(dim_val, 2)} \\text{{ м}}"
                ze_expr = f"z_e = h = {fmt_num(h_max, 3)} \\text{{ м}}"
                ze_zone_info = "для всей высоты здания"
            else:
                h_cond_str = f"{dim_code} = {fmt_num(dim_val, 2)} \\text{{ м}} < h = {fmt_num(h_max, 3)} \\text{{ м}} \\le 2 \\cdot {dim_code} = {fmt_num(2*dim_val, 2)} \\text{{ м}}"
                if z_sel <= h_max - dim_val:
                    ze_expr = f"z_e = h - {dim_code} = {fmt_num(h_max - dim_val, 3)} \\text{{ м}}"
                    ze_zone_info = f"нижняя зона: $z \\le {fmt_num(h_max - dim_val, 3)}$ м"
                else:
                    ze_expr = f"z_e = h = {fmt_num(h_max, 3)} \\text{{ м}}"
                    ze_zone_info = f"верхняя зона: $z > {fmt_num(h_max - dim_val, 3)}$ м"

            st.markdown(f"""
            #### Шаг 1. Геометрическое условие СП 20.13330.2016 (п. 11.1.5)
            * Высота здания $h = \\max(z) = {fmt_num(h_max, 3)}$ м
            * Габариты здания: ширина $d = {fmt_num(d, 2)}$ м, длина $l = {fmt_num(l, 2)}$ м
            * **Привязка зоны {selected_zone_name}**: сторона **{dim_label}**
            * Условие по высотности для определяющего размера ${dim_code} = {fmt_num(dim_val, 2)}$ м: ${h_cond_str}$
            * Эквивалентная высота для отметки $z = {fmt_num(z_sel, 3)}$ м: ${ze_expr}$ ({ze_zone_info})

            #### Шаг 2. Константы Таблицы 11.3 для местности типа '{terrain}'
            * $k_{{10}} = {k10}$
            * $\\alpha = {alpha} \\implies 2\\alpha = {two_alpha:.2f}$
            * Расчётное значение высоты $z_{{eff}} = {fmt_num(ze_eff, 3)}$ м

            #### Шаг 3. Расчёт $k(z_e)$ по формуле (11.4)
            $$k(z_e) = k_{{10}} \\cdot \\left(\\frac{{z_{{eff}}}}{{10}}\\right)^{{2\\alpha}} = {k10} \\cdot \\left(\\frac{{{fmt_num(ze_eff, 3)}}}{{10}}\\right)^{{{two_alpha:.2f}}} = \\mathbf{{{fmt_num(k_ze_val, 4)}}}$$

            #### Шаг 4. Средняя $w_m$ и расчётная $w_d$ ветровая нагрузка
            * **Нормативное давление $w_0$**: $w_0 = {w0}\\text{{ кПа}}$ *(по ветровому району **{region}**)*
            * **Учитываемый размер стороны**: ${dim_code} = {fmt_num(dim_val, 2)}\\text{{ м}}$ *(для стороны **{dim_code}**)*
            * **Аэродинамический коэффициент $c$**: $c = {c_val}$ *(по выбранной зоне **{selected_zone_name}**)*
            * **Коэффициент надёжности по нагрузке $\\gamma_f$**: $\\gamma_f = {gamma_f}$
            * **Средняя составляющая ветровой нагрузки $w_m$**:
              $$w_m = w_0 \\cdot k(z_e) \\cdot c = {w0} \\cdot {fmt_num(k_ze_val, 4)} \\cdot ({c_val}) = \\mathbf{{{fmt_num(wm_val, 4)}\\text{{ кПа}}}}$$
            * **Расчётная ветровая нагрузка $w_d$**:
              $$w_d = w_m \\cdot \\gamma_f = {fmt_num(wm_val, 4)} \\cdot {gamma_f} = \\mathbf{{{fmt_num(wd_val, 4)}\\text{{ кПа}}}} \\quad (\\approx {fmt_num(wd_val * 101.97, 1)}\\text{{ кгс/м}}^2)$$
            """)

# ------------------------------------------
# ВКЛАДКА 4: Эпюры и графики
# ------------------------------------------
with tab4:
    if not st.session_state.calculated:
        st.info("💡 Нажмите **«▶️ Выполнить расчёт»**, чтобы построить эпюру.")
    else:
        st.subheader("Эпюра эквивалентной высоты $z_e$ и расчётного давления $w_d$")
        zones_str = st.session_state.df_zones['Зона'].astype(str).tolist() if not st.session_state.df_zones.empty else []
        c_IV = st.session_state.df_zones.loc[st.session_state.df_zones['Зона'].astype(str) == 'IV', 'c'].values[0] if 'IV' in zones_str else 0.8
        
        z_plot = levels_list
        ze_plot = []
        wd_plot_d = []
        for z_v in z_plot:
            ze_v, _ = get_ze_for_height(z_v, h_max, d)
            k_ze_v = get_k_z(ze_v, terrain)
            wd_v = w0 * k_ze_v * c_IV * gamma_f
            ze_plot.append(ze_v)
            wd_plot_d.append(wd_v)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ze_plot, y=z_plot, mode='lines+markers', name='z_e (d=ширина) [м]', line=dict(color='orange', width=3)))
        fig.add_trace(go.Scatter(x=wd_plot_d, y=z_plot, mode='lines+markers', name=f'Wd (Зона IV, c={c_IV}) [кПа]', line=dict(color='navy', width=3)))
        fig.update_layout(xaxis_title="Значение (м / кПа)", yaxis_title="Отметка высоты z (м)", height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# ВКЛАДКА 5: Нагрузка с учётом грузовой площади (Wp)
# ------------------------------------------
with tab5:
    if not st.session_state.calculated:
        st.info("💡 Нажмите **«▶️ Выполнить расчёт»**, чтобы получить таблицу нагрузок с учётом грузовой площади.")
    else:
        h_grp_dict = calculate_h_grp(raw_levels_list)
        
        table_wp_data = []
        subheaders_t5 = {
            "№": "—",
            "Отметка z (м)": "—",
            "h_гр (м)": "—",
        }

        for _, z_row in st.session_state.df_zones.iterrows():
            zone_name = str(z_row['Зона']) if pd.notna(z_row['Зона']) else ""
            side_str = str(z_row.get('Сторона', 'd')) if pd.notna(z_row.get('Сторона')) else 'd'
            dim_val, _, dim_code = get_zone_dim(side_str, d, l)
            subheaders_t5[f"Зона {zone_name} (кН/м)"] = f"{dim_code} = {fmt_num(dim_val, 2)} м"

        for idx, z_val in enumerate(levels_list, 1):
            h_grp = h_grp_dict.get(z_val, 0.0)
            
            row = {
                "№": int(idx),
                "Отметка z (м)": fmt_num(z_val, 3),
                "h_гр (м)": fmt_num(h_grp, 3),
            }
            for _, z_row in st.session_state.df_zones.iterrows():
                zone_name = str(z_row['Зона']) if pd.notna(z_row['Зона']) else ""
                side_str = str(z_row.get('Сторона', 'd')) if pd.notna(z_row.get('Сторона')) else 'd'
                c_val = float(z_row['c']) if pd.notna(z_row['c']) else 0.0
                
                dim_val, _, _ = get_zone_dim(side_str, d, l)
                ze_zone, _ = get_ze_for_height(z_val, h_max, dim_val)
                k_ze = get_k_z(ze_zone, terrain)
                
                wm = w0 * k_ze * c_val
                wp = wm * gamma_f * h_grp
                
                row[f"Зона {zone_name} (кН/м)"] = fmt_num(wp, 3)
            table_wp_data.append(row)
            
        df_wp_result = pd.DataFrame(table_wp_data)

        # Заголовок и кнопка экспорта справа
        col_h5, col_exp5 = st.columns([3, 1])
        with col_h5:
            st.subheader("Таблица ветровых нагрузок с учётом грузовой площади $W_p$ (кН/м)")
        with col_exp5:
            csv_bytes_t5 = convert_df_to_csv(df_wp_result)
            st.download_button(
                label="📥 Скачать в CSV (.csv)",
                data=csv_bytes_t5,
                file_name="ветровая_нагрузка_Wp.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download_t5"
            )

        st.markdown("**Формула расчёта:**")
        st.latex(r"W_p = w_m \cdot \gamma_f \cdot h_{\text{гр}} = w_d \cdot h_{\text{гр}} \quad [\text{кН/м}]")
        st.caption("Примечание: Для отметки 0.000 м расчёт не производится.")

        render_custom_table(df_wp_result, subheaders_t5)


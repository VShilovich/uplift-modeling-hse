import json
import requests
import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000"
ADMIN_USERNAME_DEFAULT = "myadmin"
ADMIN_PASSWORD_DEFAULT = "mypass123"

st.set_page_config(page_title="Uplift API Dashboard", layout="wide")

# --- Session state ---
for key, default in {
    "jwt_token": "",
    "logged_in": False,
    "login_error": "",
    "last_request": None,
    "last_response": None,
    "last_client_df": None,
}.items():
    st.session_state.setdefault(key, default)


def api_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if st.session_state.jwt_token:
        headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
    return headers


# --- Sidebar: логин ---
st.sidebar.markdown("## 🔐 Админ‑панель")
st.sidebar.markdown("Войдите под админом, чтобы видеть историю, статистику и управлять пользователями.")

with st.sidebar.form("login_form"):
    username = st.text_input("Логин", value=ADMIN_USERNAME_DEFAULT)
    password = st.text_input("Пароль", type="password", value=ADMIN_PASSWORD_DEFAULT)
    login_btn = st.form_submit_button("Войти")

if login_btn:
    try:
        resp = requests.post(
            f"{API_URL}/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.jwt_token = data["access_token"]
            st.session_state.logged_in = True
            st.session_state.login_error = ""
            st.sidebar.success("Логин успешен")
        else:
            st.session_state.logged_in = False
            st.session_state.jwt_token = ""
            st.session_state.login_error = f"{resp.status_code}: {resp.text}"
            st.sidebar.error("Неверный логин или пароль")
    except Exception as e:
        st.session_state.logged_in = False
        st.session_state.jwt_token = ""
        st.session_state.login_error = str(e)
        st.sidebar.error("Ошибка запроса к API")

if st.session_state.logged_in:
    st.sidebar.success("Доступ к админским разделам открыт")
else:
    st.sidebar.info("Без логина доступен только инференс")

st.title("Uplift API — Дашборд")

tab_infer, tab_history, tab_stats, tab_admins = st.tabs(
    ["🎯 Инференс", "📜 История", "📈 Статистика", "👤 Админы"]
)

# ---------- Инференс (/forward) ----------
with tab_infer:
    st.header("Инференс через /forward")

    mode = st.radio(
        "Выберите способ подачи данных",
        ["Ручной ввод", "Загрузка CSV"],
        horizontal=True,
    )

    def build_payload_from_manual():
        st.subheader("Клиенты (client)")
        client_df = st.data_editor(
            pd.DataFrame(
                [
                    {
                        "client_id": 123,
                        "age": 35,
                        "gender": "F",
                        "first_issue_date": "2022-01-10",
                        "first_redeem_date": None,
                    }
                ]
            ),
            num_rows="dynamic",
            key="client_editor",
        )

        st.subheader("Покупки (purchases)")
        purchases_df = st.data_editor(
            pd.DataFrame(
                [
                    {
                        "client_id": 123,
                        "transaction_id": 1,
                        "transaction_datetime": "2024-02-01 12:30:00",
                        "purchase_sum": 540,
                        "store_id": "54a4a11a29",
                        "regular_points_received": 20,
                        "express_points_received": 0,
                        "regular_points_spent": 0,
                        "express_points_spent": 0,
                        "product_id": "9a80204f78",
                        "product_quantity": 2,
                        "trn_sum_from_iss": 540,
                        "trn_sum_from_red": 0,
                    }
                ]
            ),
            num_rows="dynamic",
            key="purchases_editor",
        )

        payload = {
            "client": client_df.to_dict(orient="records"),
            "purchases": purchases_df.to_dict(orient="records"),
        }
        return payload, client_df

    def build_payload_from_csv():
        st.subheader("Загрузка CSV")
        client_file = st.file_uploader(
            "Файл клиентов (client*.csv)", type=["csv"], key="client_csv"
        )
        purchases_file = st.file_uploader(
            "Файл покупок (purchases*.csv)", type=["csv"], key="purchases_csv"
        )

        client_df = None
        purchases_df = None

        if client_file is not None:
            client_df = pd.read_csv(client_file)
            st.write("Превью client:")
            st.dataframe(client_df.head())

        if purchases_file is not None:
            purchases_df = pd.read_csv(purchases_file)
            st.write("Превью purchases:")
            st.dataframe(purchases_df.head())

        if client_df is None or purchases_df is None:
            st.info("Загрузите оба файла, чтобы сформировать запрос.")
            return None, None

        if "client_id" not in client_df.columns or "client_id" not in purchases_df.columns:
            st.error("В обоих CSV должен быть столбец 'client_id'.")
            return None, None

        payload = {
            "client": client_df.to_dict(orient="records"),
            "purchases": purchases_df.to_dict(orient="records"),
        }
        return payload, client_df

    with st.form("uplift_form"):
        if mode == "Ручной ввод":
            payload, client_df_local = build_payload_from_manual()
        else:
            payload, client_df_local = build_payload_from_csv()

        submitted = st.form_submit_button("Запросить uplift")

    if submitted:
        if payload is None:
            st.error("Нет данных для отправки.")
        else:
            try:
                resp = requests.post(
                    f"{API_URL}/forward",
                    json=payload,
                    headers=api_headers(),
                    timeout=30,
                )
                if resp.status_code == 200:
                    resp_json = resp.json()
                    st.session_state.last_request = payload
                    st.session_state.last_response = resp_json
                    st.session_state.last_client_df = client_df_local
                    st.success("Предикты получены")
                else:
                    st.error(f"/forward вернул {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Ошибка запроса к /forward: {e}")

    st.subheader("Результаты по клиентам")

    df_upl = None
    if st.session_state.last_response is not None:
        upl = st.session_state.last_response.get("uplift", [])
        if upl:
            df_upl = pd.DataFrame(upl)
            # храним только uplift в процентах
            if "uplift" in df_upl.columns:
                df_upl["uplift_percentage"] = (df_upl["uplift"] * 100).round(4)
                df_upl = df_upl.drop(columns=["uplift"])
            st.dataframe(df_upl, use_container_width=True, hide_index=True)
        else:
            st.info("В ответе нет поля 'uplift'.")
    else:
        st.write("Сначала отправьте запрос, чтобы увидеть результаты.")

    st.subheader("Аналитика по признакам клиента и uplift (%)")

    if df_upl is not None and st.session_state.last_client_df is not None:
        client_df_all = st.session_state.last_client_df.copy()
        if "client_id" in client_df_all.columns and "client_id" in df_upl.columns:
            client_df_all["client_id"] = client_df_all["client_id"].astype(
                df_upl["client_id"].dtype
            )
        merged = client_df_all.merge(df_upl, on="client_id", how="left")

        st.write("Объединённые данные:")
        st.dataframe(merged, use_container_width=True, hide_index=True)

        # признаки клиента (как есть в client_df)
        client_cols = list(st.session_state.last_client_df.columns)
        group_options = client_cols  # client_id + все, что характеризует клиента

        group_cols = st.multiselect(
            "Признаки клиента для группировки",
            group_options,
            default=["client_id"] if "client_id" in group_options else group_options[:1],
        )

        if group_cols:
            if "uplift_percentage" in merged.columns:
                stats = (
                    merged.groupby(group_cols)["uplift_percentage"]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .round(4)
                    .reset_index()
                )

                st.subheader("Сводная таблица uplift по группам")
                st.dataframe(stats, use_container_width=True, hide_index=True)
            else:
                st.error("Нет uplift_percentage в данных.")
        else:
            st.caption("Выбери хотя бы один признак для группировки.")
    else:
        st.caption("Для аналитики нужны и client‑данные, и uplift — сначала сделай инференс.")


# ---------- История (/history) ----------
with tab_history:
    st.header("История запросов (/history)")

    if not st.session_state.logged_in:
        st.warning("Войдите как админ, чтобы смотреть историю.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            refresh_hist = st.button("Обновить историю")
        with col2:
            clear_hist = st.button("Очистить историю")

        if refresh_hist:
            try:
                resp = requests.get(
                    f"{API_URL}/history",
                    headers=api_headers(),
                    timeout=20,
                )
                if resp.status_code == 200:
                    hist = resp.json()
                    if hist:
                        df_hist = pd.DataFrame(hist)
                        st.subheader("Последние запросы")
                        st.dataframe(
                            df_hist[
                                [
                                    "id",
                                    "timestamp",
                                    "processing_time",
                                    "input_size",
                                    "input_tokens",
                                    "status",
                                ]
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.info("История пуста.")
                else:
                    st.error(f"/history вернул {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Ошибка запроса к /history: {e}")

        if clear_hist:
            try:
                resp = requests.delete(
                    f"{API_URL}/history",
                    headers=api_headers(),
                    timeout=10,
                )
            except Exception as e:
                st.error(f"Ошибка запроса к /history [DELETE]: {e}")
            else:
                if resp.status_code == 200:
                    st.success("История очищена.")
                else:
                    st.error(f"DELETE /history вернул {resp.status_code}: {resp.text}")

# ---------- Статистика (/stats) ----------
with tab_stats:
    st.header("Статистика запросов (/stats)")

    if not st.session_state.logged_in:
        st.warning("Войдите как админ, чтобы смотреть статистику.")
    else:
        if st.button("Обновить статистику"):
            try:
                resp = requests.get(
                    f"{API_URL}/stats",
                    headers=api_headers(),
                    timeout=10,
                )
                if resp.status_code == 200:
                    stats = resp.json()

                    pt = stats.get("processing_time", {})
                    ic = stats.get("input_characteristics", {})
                    size = ic.get("input_size_bytes", {})
                    tokens = ic.get("input_tokens", {})

                    st.subheader("Время обработки")
                    df_pt = pd.DataFrame(
                        {
                            "метрика": ["mean", "p50", "p95", "p99", "count", "total"],
                            "значение": [
                                pt.get("mean"),
                                pt.get("p50"),
                                pt.get("p95"),
                                pt.get("p99"),
                                pt.get("count"),
                                pt.get("total"),
                            ],
                        }
                    )
                    st.table(df_pt)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Размер входа (байты)")
                        df_size = pd.DataFrame(
                            {
                                "метрика": ["mean", "total", "count"],
                                "значение": [
                                    size.get("mean"),
                                    size.get("total"),
                                    size.get("count"),
                                ],
                            }
                        )
                        st.table(df_size)
                    with col2:
                        st.subheader("Количество токенов")
                        df_tokens = pd.DataFrame(
                            {
                                "метрика": ["mean", "total", "count"],
                                "значение": [
                                    tokens.get("mean"),
                                    tokens.get("total"),
                                    tokens.get("count"),
                                ],
                            }
                        )
                        st.table(df_tokens)
                else:
                    st.error(f"/stats вернул {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Ошибка запроса к /stats: {e}")

# ---------- Админы (/admins + чтение из SQLite) ----------
with tab_admins:
    st.header("Управление администраторами")

    if not st.session_state.logged_in:
        st.warning("Войдите как админ, чтобы управлять пользователями.")
    else:
        st.subheader("Список админов")

        try:
            resp = requests.get(
                f"{API_URL}/admins",
                headers=api_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                admins = resp.json()
                if admins:
                    df_admins = pd.DataFrame(admins)
                    st.dataframe(df_admins, use_container_width=True, hide_index=True)
                else:
                    st.info("В таблице admins пока пусто.")
            else:
                st.error(f"GET /admins вернул {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Не удалось прочитать admins через API: {e}")

        st.subheader("Создать нового админа")

        with st.form("create_admin_form"):
            new_username = st.text_input("Новый логин")
            new_password = st.text_input("Пароль", type="password")
            create_btn = st.form_submit_button("Создать админа")

        if create_btn:
            if not new_username or not new_password:
                st.error("Нужно ввести логин и пароль.")
            else:
                try:
                    resp = requests.post(
                        f"{API_URL}/admins",
                        json={"username": new_username, "password": new_password},
                        headers=api_headers(),
                        timeout=10,
                    )
                except Exception as e:
                    st.error(f"Ошибка запроса к /admins: {e}")
                else:
                    if resp.status_code == 200:
                        st.success("Админ создан или уже существовал.")
                    else:
                        st.error(f"/admins вернул {resp.status_code}: {resp.text}")
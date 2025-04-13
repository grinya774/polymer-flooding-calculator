# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf
import plotly.graph_objs as go
import io


def user_inputs():
    st.title("Калькулятор полимерного заводнения")
    st.sidebar.header("Входные параметры")
    distance_between_wells = st.sidebar.number_input("Расстояние между скважинами (м)", value=300.0, min_value=1.0)
    thickness = st.sidebar.number_input("Толщина пласта (м)", value=18.7, min_value=0.1)
    width = st.sidebar.number_input("Ширина пласта (м)", value=150.0, min_value=1.0)
    porosity = st.sidebar.number_input("Пористость (дол. ед.)", value=0.2, min_value=0.01, max_value=1.0)
    polymer_concentration = st.sidebar.number_input("Концентрация полимера (%)", value=0.46, min_value=0.0)
    injection_rate = st.sidebar.number_input("Темп закачки (м³/сут)", value=350.0, min_value=1.0)
    oil_price_per_m3 = st.sidebar.number_input("Цена нефти (руб/м³)", value=5000.0, min_value=0.0)
    polymer_cost = st.sidebar.number_input("Стоимость полимера (руб/кг)", value=150.0, min_value=0.0)
    return distance_between_wells, thickness, width, porosity, polymer_concentration, injection_rate, oil_price_per_m3, polymer_cost


def calculate_velocity(injection_rate, width, thickness):
    try:
        return injection_rate / (width * thickness)
    except ZeroDivisionError:
        return 0.0


def calculate_mixing_velocity(filtration_speed, porosity):
    try:
        return filtration_speed / porosity
    except ZeroDivisionError:
        return 0.0


def calculate_time_to_reach(distance_between_wells, injection_rate, width, thickness, porosity):
    filtration_speed = calculate_velocity(injection_rate, width, thickness)
    mixing_velocity = calculate_mixing_velocity(filtration_speed, porosity)
    return distance_between_wells / mixing_velocity if mixing_velocity > 0 else float('inf')


def calculate_polymer_concentration(x, t, vm, C0, y=None, width=150, D=1e-9):
    base_conc = C0 / 2 * (1 - erf((x - vm * t) / np.sqrt(4 * D * t))) if t > 0 else np.zeros_like(x)
    scaled_conc = base_conc * 0.5 / C0
    if y is not None:
        y_center = width / 2
        sigma = width / 4
        gaussian = np.exp(-((y - y_center) ** 2) / (2 * sigma ** 2))
        scaled_conc = scaled_conc * gaussian
    return scaled_conc * (1 + 0.5 * np.sin(2 * np.pi * x / 100))


def calculate_adsorption(C, qm=1.0, b=0.1):
    return qm * b * C / (1 + b * C)


def calculate_eor_incremental_recovery(porosity, thickness, width, distance_between_wells, sweep_efficiency=0.5,
                                       heterogeneity_factor=0.8):
    porosity_factor = porosity
    thickness_factor = thickness
    width_factor = np.log1p(width)
    reservoir_volume = porosity_factor * thickness_factor * width_factor * distance_between_wells
    oil_recovery_volume = reservoir_volume * sweep_efficiency * heterogeneity_factor
    return oil_recovery_volume


def calculate_net_profit(oil_recovery_m3, oil_price_per_m3, polymer_concentration, injection_rate, polymer_cost=150,
                         days=30):
    oil_revenue = oil_recovery_m3 * oil_price_per_m3
    polymer_mass = polymer_concentration / 100 * injection_rate * 1000 * days
    polymer_expense = polymer_mass * polymer_cost
    net_profit = oil_revenue - polymer_expense
    return max(net_profit, 0)


def sensitivity_analysis(params, ranges, base_oil):
    results = []
    change_factors = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])

    for param in ranges:
        p_base = params[param]
        for factor in change_factors:
            temp_params = params.copy()
            new_value = p_base * (1 + factor)
            if param == "porosity":
                new_value = max(0.01, min(1.0, new_value))
            temp_params[param] = new_value
            oil = calculate_eor_incremental_recovery(**temp_params)
            p_change = factor
            oil_change = (oil - base_oil) / base_oil if base_oil > 0 else 0
            results.append({"Параметр": param, "Изменение параметра (%)": p_change, "Изменение добычи (%)": oil_change})
    return pd.DataFrame(results)


def compare_with_tnavigator(oil_recovery_m3, tnavigator_oil=349000):
    return {"Python": oil_recovery_m3, "tNavigator": tnavigator_oil,
            "Разница (%)": (oil_recovery_m3 - tnavigator_oil) / tnavigator_oil * 100}


def main():
    st.set_page_config(page_title="Калькулятор полимерного заводнения")

    # Входные данные
    inputs = user_inputs()
    distance_between_wells, thickness, width, porosity, polymer_concentration, injection_rate, oil_price_per_m3, polymer_cost = inputs

    # Расчеты
    filtration_speed = calculate_velocity(injection_rate, width, thickness)
    mixing_velocity = calculate_mixing_velocity(filtration_speed, porosity)
    time_days = calculate_time_to_reach(distance_between_wells, injection_rate, width, thickness, porosity)
    oil_recovery_m3 = calculate_eor_incremental_recovery(porosity, thickness, width, distance_between_wells)
    net_profit = calculate_net_profit(oil_recovery_m3, oil_price_per_m3, polymer_concentration, injection_rate,
                                      polymer_cost)
    adsorption = calculate_adsorption(polymer_concentration / 100)

    # Вывод метрик списком
    st.markdown("### Результаты")
    st.write(f"Скорость фильтрации: {filtration_speed:.2f} м/сут")
    st.write(f"Скорость смешения: {mixing_velocity:.2f} м/сут")
    st.write(f"Время достижения: {time_days:.2f} сут")
    st.write(f"Дополнительная добыча: {oil_recovery_m3:.2f} м³")
    st.write(f"Чистая прибыль: {net_profit:.2f} руб")
    st.write(f"Адсорбция полимера: {adsorption:.4f} кг/м³")

    # Экспорт таблицы результатов
    results_df = pd.DataFrame({
        'Показатель': ['Скорость фильтрации (м/сут)', 'Скорость смешения (м/сут)', 'Время достижения (сут)',
                       'Дополнительная добыча (м³)', 'Чистая прибыль (руб)', 'Адсорбция полимера (кг/м³)'],
        'Значение': [filtration_speed, mixing_velocity, time_days, oil_recovery_m3, net_profit, adsorption]
    })
    if st.button("Сохранить результаты в Excel", key="save_excel"):
        with io.BytesIO() as excel_buffer:
            results_df.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_buffer.seek(0)
            st.download_button(
                label="Сохранить результаты в Excel",
                data=excel_buffer,
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )

    # Визуализации
    st.markdown("### Визуализации")
    tab1, tab2, tab3 = st.tabs(["Анализ чувствительности", "3D распределение", "Насыщенность и адсорбция"])

    # Вкладка 1: Анализ чувствительности
    with tab1:
        st.markdown("#### Анализ чувствительности")
        ranges = {
            "porosity": np.linspace(0.15, 0.25, 5),
            "thickness": np.linspace(10, 25, 5),
            "width": np.linspace(100, 200, 5)
        }
        params = {"porosity": porosity, "thickness": thickness, "width": width,
                  "distance_between_wells": distance_between_wells}
        sens_df = sensitivity_analysis(params, ranges, oil_recovery_m3)
        st.write("Данные для porosity:")
        st.write(sens_df[sens_df["Параметр"] == "porosity"])
        fig_sens, ax_sens = plt.subplots(figsize=(8, 6))
        line_styles = {"porosity": "-", "thickness": "--", "width": ":"}
        colors = {"porosity": "blue", "thickness": "green", "width": "red"}
        for param in ranges:
            subset = sens_df[sens_df["Параметр"] == param]
            if not subset.empty:
                ax_sens.plot(subset["Изменение параметра (%)"] * 100, subset["Изменение добычи (%)"] * 100,
                             label=param, linestyle=line_styles[param], color=colors[param], marker='o', linewidth=2)
            else:
                st.warning(f"Нет данных для параметра {param}")
        ax_sens.set_xlabel("Относительное изменение параметра (%)")
        ax_sens.set_ylabel("Относительное изменение добычи (%)")
        ax_sens.legend()
        ax_sens.grid(True)
        ax_sens.axhline(0, color='black', linewidth=0.5)
        ax_sens.axvline(0, color='black', linewidth=0.5)
        ax_sens.set_xlim(-25, 25)
        ax_sens.set_ylim(-50, 50)
        st.pyplot(fig_sens)
        if st.button("Сохранить график анализа чувствительности в JPEG", key="save_sens"):
            with io.BytesIO() as img_buffer:
                fig_sens.savefig(img_buffer, format="jpeg", bbox_inches="tight")
                img_buffer.seek(0)
                st.download_button(
                    label="Сохранить график анализа чувствительности в JPEG",
                    data=img_buffer,
                    file_name="sensitivity_plot.jpeg",
                    mime="image/jpeg",
                    key="download_sens"
                )

    # Вкладка 2: 3D распределение
    with tab2:
        st.markdown("#### 3D распределение полимера")
        x = np.linspace(0, distance_between_wells, 100)
        y = np.linspace(0, width, 100)
        X, Y = np.meshgrid(x, y)
        Z = calculate_polymer_concentration(X, time_days * 24 * 3600, mixing_velocity * 24 * 3600,
                                            polymer_concentration / 100, Y, width)
        Z = Z * 100
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Plasma')])
        fig_3d.update_layout(scene=dict(
            xaxis_title='Расстояние (м)',
            yaxis_title='Ширина (м)',
            zaxis_title='Концентрация (%)'
        ))
        st.plotly_chart(fig_3d)
        if st.button("Сохранить 3D график в JPEG", key="save_3d"):
            with io.BytesIO() as img_buffer:
                fig_3d.write_image(img_buffer, format="jpeg", engine="kaleido")
                img_buffer.seek(0)
                st.download_button(
                    label="Сохранить 3D график в JPEG",
                    data=img_buffer,
                    file_name="3d_plot.jpeg",
                    mime="image/jpeg",
                    key="download_3d"
                )

    # Вкладка 3: Тепловая карта
    with tab3:
        st.markdown("#### Тепловая карта насыщенности и адсорбции")
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
        x = np.linspace(0, distance_between_wells, 50)
        y = np.linspace(0, width, 50)
        X, Y = np.meshgrid(x, y)
        polymer_saturation = calculate_polymer_concentration(X, time_days * 24 * 3600, mixing_velocity * 24 * 3600,
                                                             polymer_concentration / 100, Y, width)
        heatmap_data = polymer_saturation
        vmin, vmax = polymer_saturation.min(), polymer_saturation.max()
        sns.heatmap(heatmap_data, cmap='RdYlBu', ax=ax_heatmap, annot=False, vmin=vmin, vmax=vmax)
        ax_heatmap.set_xlabel('Расстояние (м)')
        ax_heatmap.set_ylabel('Ширина (м)')
        ax_heatmap.set_title('Насыщенность полимером (дол. ед.)')
        st.pyplot(fig_heatmap)
        adsorption_map = calculate_adsorption(polymer_saturation)
        st.write(f"Средняя адсорбция (кг/м³): {adsorption_map.mean():.4f}")
        if st.button("Сохранить тепловую карту в JPEG", key="save_heatmap"):
            with io.BytesIO() as img_buffer:
                fig_heatmap.savefig(img_buffer, format="jpeg", bbox_inches="tight")
                img_buffer.seek(0)
                st.download_button(
                    label="Сохранить тепловую карту в JPEG",
                    data=img_buffer,
                    file_name="heatmap.jpeg",
                    mime="image/jpeg",
                    key="download_heatmap"
                )


if __name__ == "__main__":
    main()
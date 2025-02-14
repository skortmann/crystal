import matplotlib.pyplot as plt

plt.style.use(["rwth-latex"])
from rwth_colors import colors
import numpy as np
import pandas as pd
import os

# get results and save each as png file

color1 = (0 / 255, 84 / 255, 159 / 255, 1)
color2 = (227 / 255, 0 / 255, 102 / 255, 1)
color3 = (87 / 255, 171 / 255, 39 / 255, 1)


def create_and_save_plots(
    daa_price_vector,
    ida_price_vector,
    idc_price_vector,
    revenue_total,
    revenue_daa_today,
    revenue_ida_today,
    revenue_idc_today,
    step1_cha_daa,
    step1_dis_daa,
    step2_cha_ida,
    step2_dis_ida,
    step2_cha_ida_close,
    step2_dis_ida_close,
    step3_cha_idc,
    step3_dis_idc,
    step3_cha_idc_close,
    step3_dis_idc_close,
    step3_soc_idc,
    save_directory,
    filename,
):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    xgrid = np.arange(1, len(daa_price_vector) + 1, 1)

    fig, axs = plt.subplots(
        5, 1, figsize=(9, 12), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]}
    )

    ax1, ax2, ax3, ax4, ax5 = axs

    plot_1(
        xgrid,
        ax1,
        daa_price_vector,
        ida_price_vector,
        idc_price_vector,
        revenue_total,
        revenue_daa_today,
        revenue_ida_today,
        revenue_idc_today,
    )
    plot_2(xgrid, ax2, daa_price_vector, step1_cha_daa, step1_dis_daa)
    plot_3(
        xgrid,
        ax3,
        ida_price_vector,
        step2_cha_ida,
        step2_dis_ida,
        step2_cha_ida_close,
        step2_dis_ida_close,
    )
    plot_4(
        xgrid,
        ax4,
        idc_price_vector,
        step3_cha_idc,
        step3_dis_idc,
        step3_cha_idc_close,
        step3_dis_idc_close,
    )
    plot_5(xgrid, ax5, ida_price_vector, step3_soc_idc)

    plt.tight_layout()
    file_path = os.path.join(save_directory, filename)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(file_path)
    plt.close()


# Subplot 1 --> Preisverlauf auf den einzelnen Märkten an einem Tag
def plot_1(
    xgrid,
    ax1,
    daa_price_vector,
    ida_price_vector,
    idc_price_vector,
    revenue_total,
    revenue_daa_today,
    revenue_ida_today,
    revenue_idc_today,
):
    ax1.plot(xgrid, daa_price_vector, color=color1, label="DA Auction price")
    ax1.plot(xgrid, ida_price_vector, color=color2, label="ID Auction price")
    ax1.plot(xgrid, idc_price_vector, color=color3, label="IDC ID1 price")

    ax1.legend(ncol=4, loc=(0.15, 0.85))
    ax1.grid()

    ax1.set_xlim(1, len(daa_price_vector))
    xticks = np.linspace(1, len(daa_price_vector), 9, dtype=int)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks)

    yticks = [-50, -25, 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks)
    ax1.set_ylim(-60, 280)
    ax1.set_ylabel("Prices \n [EUR/MW]")
    ax1.set_title(
        f"Example year cross market positions \n Total revenue: {revenue_total:.2f} EUR \n Today: DAA-Revenue: {revenue_daa_today:.2f}, IDA-Revenue: {revenue_ida_today:.2f}, IDC-Revenue: {revenue_idc_today:.2f}"
    )


# Subplot 2 --> Positionen in MWh, die bei DA-Auktionen eingegangen wurden
def plot_2(xgrid, ax2, daa_price_vector, step1_cha_daa, step1_dis_daa):
    ax2.bar(
        xgrid,
        np.asarray(step1_cha_daa) / 4,
        color=color1,
        label="Positions on DA Auction",
        linewidth=1.75,
    )  # bar() für ein Balkendiagramm
    ax2.bar(xgrid, np.asarray(step1_dis_daa) * -1 / 4, color=color1, linewidth=1.75)

    ax2.legend(ncol=4, loc=(0.355, 0.75))
    ax2.grid()

    ax2.set_xlim(1, len(daa_price_vector))
    xticks = np.linspace(1, len(daa_price_vector), 9, dtype=int)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticks)

    ax2.set_ylim(-0.3, 0.5)
    ax2.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax2.set_yticklabels(ax2.get_yticks())
    ax2.set_ylabel("Positions \n [MWh]")


# Subplot 3 --> Positionen in MWh, die bei ID-Auktionen eingegangen wurden
def plot_3(
    xgrid,
    ax3,
    ida_price_vector,
    step2_cha_ida,
    step2_dis_ida,
    step2_cha_ida_close,
    step2_dis_ida_close,
):
    ax3.bar(
        xgrid,
        np.asarray(step2_cha_ida) / 4,
        color=color2,
        label="Positions on ID Auction",
        linewidth=2.5,
    )
    ax3.bar(xgrid, np.asarray(step2_dis_ida) * -1 / 4, color=color2, linewidth=1.75)
    ax3.bar(xgrid, np.asarray(step2_cha_ida_close) / 4, color=color2, linewidth=1.75)
    ax3.bar(
        xgrid, np.asarray(step2_dis_ida_close) * -1 / 4, color=color2, linewidth=1.75
    )

    ax3.legend(ncol=4, loc=(0.365, 0.75))
    ax3.grid()

    ax3.set_xlim(1, len(ida_price_vector))
    xticks = np.linspace(1, len(ida_price_vector), 9, dtype=int)
    ax3.set_xticks(xticks)
    ax3.set_xticklabels(xticks)

    ax3.set_ylim(-0.3, 0.5)
    ax3.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax3.set_yticklabels(ax3.get_yticks())
    ax3.set_ylabel("Positions \n [MWh]")


# Subplot 4 --> Positionen in MWh, die bei ID-Continuous eingegangen wurden
def plot_4(
    xgrid,
    ax4,
    idc_price_vector,
    step3_cha_idc,
    step3_dis_idc,
    step3_cha_idc_close,
    step3_dis_idc_close,
):
    ax4.bar(
        xgrid,
        np.asarray(step3_cha_idc) / 4,
        color=color3,
        label="Positions on ID Continuous",
        linewidth=1.75,
    )
    ax4.bar(xgrid, np.asarray(step3_dis_idc) * -1 / 4, color=color3, linewidth=1.75)
    ax4.bar(xgrid, np.asarray(step3_cha_idc_close) / 4, color=color3, linewidth=1.75)
    ax4.bar(
        xgrid, np.asarray(step3_dis_idc_close) * -1 / 4, color=color3, linewidth=1.75
    )

    ax4.legend(ncol=4, loc=(0.35, 0.75))
    ax4.grid()

    ax4.set_xlim(1, len(idc_price_vector))
    xticks = np.linspace(1, len(idc_price_vector), 9, dtype=int)
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xticks)

    ax4.set_ylim(-0.3, 0.5)
    ax4.set_yticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax4.set_yticklabels(ax4.get_yticks())
    ax4.set_ylabel("Positions \n [MWh]")


# Subplot 5 --> Verlauf des State of Charges in MWh über die quarters
def plot_5(xgrid, ax5, ida_price_vector, step3_soc_idc):
    ax5.plot(
        xgrid, step3_soc_idc, color="black", label="State of charge", linewidth=1.75
    )

    ax5.legend(ncol=4, loc=(0.39, 0.75))
    ax5.grid()

    ax5.set_xlim(1, len(ida_price_vector))
    xticks = np.linspace(1, len(ida_price_vector), 9, dtype=int)
    ax5.set_xticks(xticks)
    ax5.set_xticklabels(xticks)
    ax5.set_xlabel("Time [quarters]")

    ax5.set_ylim(0, 1)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(ax5.get_yticks())
    ax5.set_ylabel("State of \n Charge \n [MWh]")


# Erstellen der Forecasting-Plots
def plot_forecasting(
    daa_price_vector_i,
    ida_price_vector_i,
    idc_price_vector_i,
    forecast_daa_filename,
    forecast_ida_filename,
    forecast_idc_filename,
    save_directory,
    filename_forecasting,
    daa_mse,
    ida_mse,
    idc_mse,
    daa_price_vector_mse,
    ida_price_vector_mse,
    idc_price_vector_mse,
):
    # Einlesen der CSV-Dateien
    df_daa = pd.read_csv(forecast_daa_filename)
    df_ida = pd.read_csv(forecast_ida_filename)
    df_idc = pd.read_csv(forecast_idc_filename)

    # Anzahl an genutzten Tagen für das Forecasting, benötigt für die Überschrift
    observed_days = len(daa_price_vector_i) // 96

    # Kürzen der price_vector_i auf die letzten 7 Tage (Observed-Werte in den Plots)
    daa_price_vector_i = daa_price_vector_i[-7 * 96 :]
    ida_price_vector_i = ida_price_vector_i[-7 * 96 :]
    idc_price_vector_i = idc_price_vector_i[-7 * 96 :]

    # Aktualisieren des xgrid basierend auf den gekürzten Vektoren
    xgrid_daa = np.arange(0, len(daa_price_vector_i), 1)
    xgrid_ida = np.arange(0, len(ida_price_vector_i), 1)
    xgrid_idc = np.arange(0, len(idc_price_vector_i), 1)

    fig, axs = plt.subplots(
        3, 1, figsize=(18 / 2, 21 / 2), gridspec_kw={"height_ratios": [1, 1, 1]}
    )
    ax_daa, ax_ida, ax_idc = axs

    # Funktion zum Plotten der Daten
    def plot_data(ax, price_vector_i, mse_vector, df, xgrid, ylabel, mse, observed):
        lower_quantile = df["0.1"].astype(float)
        higher_quantile = df["0.9"].astype(float)
        mean = df["mean"].astype(float)

        ax.grid()
        ax.set_title(
            f"Forecasting based on {observed} observed days for 1-day-Forecasting (only shows the last 7 days of Observation)"
        )
        ax.set_xlim(1, len(price_vector_i) + 96)
        xticks = np.linspace(1, len(price_vector_i) + 96, 10, dtype=int)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)
        ax.set_xlabel("quarters")
        yticks = [-100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks)
        ax.set_ylim(-100, 250)
        ax.set_ylabel(ylabel)

        # Ausgangsdaten plotten
        ax.plot(xgrid, price_vector_i, label="Observed", color="blue")

        # Forecast-Daten plotten
        hellorange = (1.0, 0.5, 0.0)
        dunkelorange = (0.8, 0.4, 0.0)

        start_index_forecast = len(price_vector_i) - 1  # Startindex für den Forecast
        x_values_forecast = range(
            start_index_forecast, start_index_forecast + len(mean)
        )
        ax.plot(
            x_values_forecast,
            mean,
            label=f"Mean Forecast (MSE: {mse:.6f})",
            color="red",
        )
        ax.plot(
            x_values_forecast,
            lower_quantile,
            label="Quantile Forecast",
            color=dunkelorange,
        )
        ax.plot(x_values_forecast, higher_quantile, color=dunkelorange)
        ax.fill_between(
            x_values_forecast,
            lower_quantile,
            higher_quantile,
            color=hellorange,
            alpha=0.3,
            label="Uncertainty Range",
        )

        # Die Werte, die dann in Zukunft wirklich eingetreten sind
        ax.plot(x_values_forecast, mse_vector, "b--", label="True future values")

        ax.legend()

    # DAA plotten
    plot_data(
        ax_daa,
        daa_price_vector_i,
        daa_price_vector_mse,
        df_daa,
        xgrid_daa,
        "Prices DAA \n [EUR/MW]",
        daa_mse,
        observed_days,
    )

    # IDA plotten
    plot_data(
        ax_ida,
        ida_price_vector_i,
        ida_price_vector_mse,
        df_ida,
        xgrid_ida,
        "Prices IDA \n [EUR/MW]",
        ida_mse,
        observed_days,
    )

    # IDC plotten
    plot_data(
        ax_idc,
        idc_price_vector_i,
        idc_price_vector_mse,
        df_idc,
        xgrid_idc,
        "Prices IDC \n [EUR/MW]",
        idc_mse,
        observed_days,
    )

    # Abspeichern
    plt.tight_layout()
    file_path = os.path.join(save_directory, filename_forecasting)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(file_path)
    plt.close()


# wenn über for-schleife
def plot_diff(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5))

    # Oberer Plot: Drei einzelne Revenues
    ax1.plot(df["Tag"], df["DAA-Revenue"], label="DAA-Revenue", color="#00549f")
    ax1.plot(df["Tag"], df["IDA-Revenue"], label="IDA-Revenue", color="#e30066")
    ax1.plot(df["Tag"], df["IDC-Revenue"], label="IDC-Revenue", color="#57ab27")

    ax1.set_xlabel("Days", fontsize=12)
    ax1.set_ylabel("Differences in revenues [€]", fontsize=12)
    ax1.set_title(
        "Cummulative difference in revenues on the individual markets", fontsize=12
    )
    ax1.legend(fontsize=12)
    ax1.grid(True)

    # x-Achse anpassen: Schritte von 50 Tagen
    ax1.set_xticks(np.arange(0, len(df) + 1, 5))

    # y-Achse für den oberen Plot anpassen
    y_min_individual = df[["DAA-Revenue", "IDA-Revenue", "IDC-Revenue"]].min().min()
    y_max_individual = df[["DAA-Revenue", "IDA-Revenue", "IDC-Revenue"]].max().max()
    ax1.set_yticks(np.arange(y_min_individual, y_max_individual + 10, 5))

    # set x-ticks to fontsize 12
    ax1.set_xticklabels([round(tick, 0) for tick in ax1.get_xticks()], fontsize=12)
    ax1.set_yticklabels([round(tick, 0) for tick in ax1.get_yticks()], fontsize=12)

    # Unterer Plot: Total Revenue
    ax2.plot(df["Tag"], df["Total-Revenue"], label="Total-Revenue", color="black")
    ax2.set_xlabel("Days", fontsize=12)
    ax2.set_ylabel("Differences in revenues [€]", fontsize=12)
    ax2.set_title("Cummulative difference in revenues", fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True)

    # x-Achse anpassen: Schritte von 50 Tagen
    ax2.set_xticks(np.arange(0, len(df) + 1, 5))

    # y-Achse für den unteren Plot anpassen
    y_max_total = df["Total-Revenue"].max()
    y_min_total = df["Total-Revenue"].min()
    ax2.set_yticks(np.arange(y_min_total - 11, y_max_total + 10, 10))

    ax2.set_xticklabels(ax2.get_xticks(), fontsize=12)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=12)

    plt.tight_layout()
    plt.savefig("Difference_Stochastic_and_Deterministic.pdf")
    plt.close()


# wenn einfach so (also kein plot über die for-Schleife)
def plot_diff_a(csv_filename, mse_csv):
    df = pd.read_csv(csv_filename)
    df_mse = pd.read_csv(mse_csv)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Berechnete Mittelwerte
    daa_mse_mean = df_mse.loc[df_mse["Market"] == "DAA", "MSE Mean"].values[0]
    ida_mse_mean = df_mse.loc[df_mse["Market"] == "IDA", "MSE Mean"].values[0]
    idc_mse_mean = df_mse.loc[df_mse["Market"] == "IDC", "MSE Mean"].values[0]

    # Oberer Plot: Drei einzelne Revenues
    ax1.plot(
        df["Tag"],
        df["DAA-Revenue"],
        label=f"DAA-Revenue (MSE Mean: {daa_mse_mean:.2f})",
        color="green",
    )
    ax1.plot(
        df["Tag"],
        df["IDA-Revenue"],
        label=f"IDA-Revenue (MSE Mean: {ida_mse_mean:.2f})",
        color="orange",
    )
    ax1.plot(
        df["Tag"],
        df["IDC-Revenue"],
        label=f"IDC-Revenue (MSE Mean: {idc_mse_mean:.2f})",
        color="blue",
    )

    ax1.set_xlabel("Tage")
    ax1.set_ylabel("Gewinndifferenz [€]")
    # ax1.set_title('Verlauf der Gewinndifferenz auf den einzelnen Märkten (Deterministisch - Stochastisch) in Euro')
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # x-Achse anpassen
    start = df["Tag"].min()
    end = df["Tag"].max()
    step = 5

    xticks = np.arange(start, end + 1, step)
    if end not in xticks:
        xticks = np.append(xticks, end)

    ax1.set_xticks(xticks)

    # y-Achse für den oberen Plot anpassen
    y_min_individual = df[["DAA-Revenue", "IDA-Revenue", "IDC-Revenue"]].min().min()
    y_max_individual = df[["DAA-Revenue", "IDA-Revenue", "IDC-Revenue"]].max().max()
    ax1.set_yticks(np.arange(y_min_individual, y_max_individual + 10, 10))

    # Unterer Plot: Total Revenue
    ax2.plot(df["Tag"], df["Total-Revenue"], label="Total-Revenue", color="red")
    ax2.set_xlabel("Tage")
    ax2.set_ylabel("Gewinndifferenz [€]")
    # ax2.set_title('Verlauf der Gesamtgewinn-Differenz (Deterministisch - Stochastisch) in Euro')
    ax2.legend()
    ax2.grid(True)

    # x-Achse anpassen
    start = df["Tag"].min()
    end = df["Tag"].max()
    step = 5

    xticks = np.arange(start, end + 1, step)
    if end not in xticks:
        xticks = np.append(xticks, end)

    ax2.set_xticks(xticks)

    # y-Achse für den unteren Plot anpassen
    y_max_total = df["Total-Revenue"].max()
    y_min_total = df["Total-Revenue"].min()
    yticks = np.arange(y_min_total, y_max_total + 10, 100)
    """
    # Sicherstellen, dass y=0 enthalten ist
    if 0 not in yticks:
        yticks = np.append(yticks, 0)

    # Sortieren der Ticks
    yticks = np.sort(yticks)

    if y_max_total not in yticks:
        yticks = np.append(yticks, y_max_total)

    yticks = np.sort(yticks)
    """
    ax2.set_yticks(yticks)

    plt.tight_layout()
    plt.savefig("aa_Difference_Stochastic_and_Deterministic")
    plt.close()


# veränderte Version
def plot_diff_b(csv_filename):
    df = pd.read_csv(csv_filename)

    # Erstelle die Figure und Subplots mit angepasstem Verhältnis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Anpassung der Höhe auf 10

    # Oberer Plot: Drei einzelne Revenues ohne MSE in der Legende
    ax1.plot(df["Tag"], df["DAA-Revenue"], label="DAA-Revenue", color="green")
    ax1.plot(df["Tag"], df["IDA-Revenue"], label="IDA-Revenue", color="orange")
    ax1.plot(df["Tag"], df["IDC-Revenue"], label="IDC-Revenue", color="blue")

    # Achsen und Legende
    ax1.set_xlabel("Tage")
    ax1.set_ylabel("Gewinndifferenz [€]")
    # ax1.legend(loc='upper left')
    ax1.legend()
    ax1.grid(True)

    # Anpassung der x-Achse
    start = df["Tag"].min()
    end = df["Tag"].max()
    step = 5
    xticks = np.arange(start, end + 1, step)
    if end not in xticks:
        xticks = np.append(xticks, end)
    ax1.set_xticks(xticks)

    # Anpassung der y-Achse im oberen Plot

    # Anpassung der y-Achse im oberen Plot
    y_min_individual = df[["DAA-Revenue", "IDA-Revenue", "IDC-Revenue"]].min().min()
    y_max_individual = df[["DAA-Revenue", "IDA-Revenue", "IDC-Revenue"]].max().max()
    ax1.set_yticks(np.arange(y_min_individual, y_max_individual + 10, 20))

    # Unterer Plot: Total Revenue ohne MSE
    ax2.plot(df["Tag"], df["Total-Revenue"], label="Total-Revenue", color="red")
    ax2.set_xlabel("Tage")
    ax2.set_ylabel("Gewinndifferenz [€]")
    ax2.legend(loc="upper left")
    ax2.grid(True)

    # x-Achse anpassen
    xticks = np.arange(start, end + 1, step)
    if end not in xticks:
        xticks = np.append(xticks, end)
    ax2.set_xticks(xticks)

    # y-Achse für den unteren Plot anpassen
    y_max_total = df["Total-Revenue"].max()
    y_min_total = df["Total-Revenue"].min()
    yticks = np.arange(y_min_total, y_max_total + 10, 250)
    ax2.set_yticks(yticks)

    # Layout anpassen und Diagramme speichern mit erhöhter DPI
    plt.tight_layout(pad=1)  # Weniger Abstand zwischen den Plots
    plt.savefig(
        "aa_Difference_Stochastic_and_Deterministic.png", dpi=300, format="png"
    )  # Speichern mit höherer Auflösung
    plt.close()


# Für den Plot, indem man den VII gegen den MSE aufträgt
def plot_VII_MSE():
    # benutzte Modelle
    models = [
        "DeepAR",
        "TemporalFusionTransformer",
        "RecursiveTabular",
        "DirectTabular",
        "Weighted Ensemble",
    ]

    # Werte aus dem Forecasting ohne zusätzliche Parameter
    mse_values = [6448.2062, 7598.6464, 5475.3403, 11098.1564, 4971.7893]
    vii_values = [249.2985, 362.2986, -971.6758, 1795.7981, -8.2007]

    # Werte aus dem Forecasting mit zusätzlichen Parametern
    mse_values_new = [5576.4893, 6870.9879, 5856.2806, 10022.3, 8530.067]
    vii_values_new = [279.9, 408.08, -946.545, 1443.79, 722.745]

    # Plot erstellen mit den alten Werten etwas blasser im Gegensatz zu den Neuen
    plt.figure(figsize=(12, 7))
    plt.scatter(
        mse_values_new,
        vii_values_new,
        color=[
            colors["blue"],
            colors["orange"],
            colors["green"],
            colors["red"],
            colors["lavender"],
        ],
    )
    plt.scatter(
        mse_values,
        vii_values,
        color=[
            colors[("blue", 50)],
            colors[("orange", 50)],
            colors[("green", 50)],
            colors[("red", 50)],
            colors[("lavender", 50)],
        ],
    )

    """''
    # Arrows hinzufügen, um die Verschiebung anzuzeigen
    for i in range(len(models)):
        plt.annotate('', xy=(mse_values_new[i], vii_values_new[i]), xytext=(mse_values[i], vii_values[i]),
                     arrowprops=dict(facecolor='black', arrowstyle='->'))
    """ ""

    # Arrows und Model-Namen hinzufügen, um die Verschiebung anzuzeigen
    for i, model in enumerate(models):
        plt.annotate(
            "",
            xy=(mse_values_new[i], vii_values_new[i]),
            xytext=(mse_values[i], vii_values[i]),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
        )

        # Dynamische Anpassung der Annotationspositionen, um Überschneidungen zu vermeiden
        offset_x = 5
        offset_y = 5
        if i % 2 == 0:
            offset_x = -10
        if i % 3 == 0:
            offset_y = -10

        plt.annotate(
            model,
            (mse_values_new[i], vii_values_new[i]),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            ha="right",
            fontsize=9,
        )

    # Achsen beschriften
    plt.xlabel("MSE in (Euro / MW) squared")
    plt.ylabel("VII in Euro")

    left, right = plt.xlim()  # return the current xlim
    plt.xlim(left * 0.9, right * 1.1)  # set the xlim to left, right

    # Legende hinzufügen an den Punkten

    for i, model in enumerate(models):
        plt.annotate(model, (mse_values[i] * 1.1, vii_values[i] * 1.1))
        plt.annotate(model, (mse_values_new[i] * 1.1, vii_values_new[i] * 1.1))

    # Legende hinzufügen in der oberen linken Ecke

    for i, model in enumerate(models):
        plt.scatter(
            [],
            [],
            color=[
                colors["blue"],
                colors["orange"],
                colors["green"],
                colors["red"],
                colors["lavender"],
            ][i],
            label=model,
        )

    # plt.legend(loc='upper left', frameon=True, fontsize='medium', markerscale=0.9)

    # plt.title('"Mean-Squared-Error" and "Value-of-Imperfect-Information" for Different Models and Parameters')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("aa_VII_and_MSE_for_different_Models_and_Parameters")
    plt.close()


# veränderte Version
def plot_VII_MSE_b():
    # benutzte Modelle
    models = [
        "DeepAR",
        "TemporalFusionTransformer",
        "RecursiveTabular",
        "DirectTabular",
        "Weighted Ensemble",
    ]

    # Werte aus dem Forecasting ohne zusätzliche Parameter
    mse_values = [6448.2062, 7598.6464, 5475.3403, 11098.1564, 4971.7893]
    vii_values = [249.2985, 362.2986, -971.6758, 1795.7981, -8.2007]

    # Werte aus dem Forecasting mit zusätzlichen Parametern
    mse_values_new = [5576.4893, 6870.9879, 5856.2806, 10022.3, 8530.067]
    vii_values_new = [279.9, 408.08, -946.545, 1443.79, 722.745]

    # Plot erstellen mit den alten Werten etwas blasser im Gegensatz zu den Neuen
    plt.figure(figsize=(12, 7))

    # Punkte plotten mit zorder höher als das Grid (z.B. 5)
    plt.scatter(
        mse_values_new,
        vii_values_new,
        color=[
            colors["blue"],
            colors["orange"],
            colors["green"],
            colors["red"],
            colors["lavender"],
        ],
        zorder=5,
    )
    plt.scatter(
        mse_values,
        vii_values,
        color=[
            colors["blue"],
            colors["orange"],
            colors["green"],
            colors["red"],
            colors["lavender"],
        ],
        alpha=0.5,
        zorder=5,
    )

    # Arrows und Model-Namen hinzufügen, um die Verschiebung anzuzeigen
    for i, model in enumerate(models):
        plt.annotate(
            "",
            xy=(mse_values_new[i], vii_values_new[i]),
            xytext=(mse_values[i], vii_values[i]),
            arrowprops=dict(facecolor="black", arrowstyle="->"),
            zorder=6,
        )  # Pfeile auch über Grid

        # Dynamische Anpassung der Annotationspositionen, um Überschneidungen zu vermeiden
        offset_x = 5
        offset_y = 5
        if i % 2 == 0:
            offset_x = -10
        if i % 3 == 0:
            offset_y = -10

        plt.annotate(
            model,
            (mse_values_new[i], vii_values_new[i]),
            textcoords="offset points",
            xytext=(offset_x, offset_y),
            ha="right",
            fontsize=9,
            zorder=6,
        )  # Modelnamen über Grid

    # Achsen beschriften
    plt.xlabel("MSE in (Euro / MW) squared")
    plt.ylabel("VII in Euro")

    # Achsenbereich anpassen
    left, right = plt.xlim()
    plt.xlim(left * 0.9, right * 1.1)

    # Legende hinzufügen in der oberen linken Ecke
    # plt.legend(models, loc='upper left', frameon=True, fontsize='medium', markerscale=0.9)

    # Titel und Gitter hinzufügen
    plt.grid(True, zorder=0)  # Grid hinter den Punkten zeichnen

    # Plot speichern mit höherer Auflösung (DPI)
    plt.tight_layout()
    plt.savefig("aa_VII_and_MSE_for_different_Models_and_Parameters", dpi=300)
    plt.close()


def BA_plot():
    import matplotlib.pyplot as plt
    import numpy as np

    # Daten
    years = ["2018", "2019", "2020", "2021"]
    day_ahead = [10000, 10000, 12000, 15000]
    id_auc = [25000, 30000, 35000, 50000]
    id_cont_id3 = [20000, 25000, 30000, 35000]
    id_auc_id_cont = [50000, 60000, 65000, 75000]

    bar_width = 0.2
    index = np.arange(len(years))

    # Balkendiagramm erstellen
    fig, ax = plt.subplots(figsize=(10, 6))

    bar1 = ax.bar(
        index - 1.5 * bar_width,
        day_ahead,
        bar_width,
        label="Day Ahead",
        color="lightgrey",
    )
    bar2 = ax.bar(
        index - 0.5 * bar_width, id_auc, bar_width, label="ID-Auc", color="lightblue"
    )
    bar3 = ax.bar(
        index + 0.5 * bar_width,
        id_cont_id3,
        bar_width,
        label="ID-Cont ID3",
        color="blue",
    )
    bar4 = ax.bar(
        index + 1.5 * bar_width,
        id_auc_id_cont,
        bar_width,
        label="ID-Auc + ID-Cont ID3",
        color="darkblue",
    )

    # Titel und Labels hinzufügen
    ax.set_ylim(0, 80000)
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Erlöse (pro MW)")
    ax.set_title(
        "Entwicklung der jährlichen Erlöse im Energiehandel mit Batteriespeichern (pro MW)"
    )
    ax.set_xticks(index)
    ax.set_xticklabels(years)
    ax.legend()

    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.7)

    # Diagramm anzeigen
    plt.tight_layout()
    plt.show()
    plt.savefig("testplot")


if __name__ == "__main__":
    # Lesen des DataFrame
    df_diff = pd.read_csv("../Difference_det_stoc_revenues.csv")

    # Plotten der Gewinndifferenzen
    plot_diff(df_diff)

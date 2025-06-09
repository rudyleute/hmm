import re
from typing import Dict, Tuple, Union

import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typeguard import typechecked

@typechecked
class Plots:
    def __init__(self):
        pass

    @staticmethod
    def showTestCasesPlot(results: Dict[str, Dict[int, Dict[Tuple, Dict[str, Dict[str, float]]]]], isGT: bool = False) -> None:
        rows = []
        for matrixType, sizeDict in results.items():
            for matrixSize, labelDict in sizeDict.items():
                for keys, nested in labelDict.items():
                    alg = nested.get("alg", {})
                    subdict = dict({
                        "PlotColumn": keys[0],
                        "Legend": keys[2]
                    })
                    if isGT:
                        subdict["PlotColumn"] = subdict["Legend"]
                        subdict["Legend"] = keys[3]

                    for metric in ["P", "A", "B"]:
                        value = alg.get(metric)
                        if value is not None:
                            rows.append({
                                "MatrixType": matrixType,
                                "Series": matrixSize,
                                "SequenceLen": keys[1],
                                "Metric": metric,
                                "Value": value,
                                **subdict
                            })

        df = pd.DataFrame(rows)
        if df["Legend"].dtype == "object":
            df = df.sort_values(by="Legend", key=lambda col: col.apply(len))

        # Choose a clearer color palette
        palette = sns.color_palette("bright", n_colors=len(df["Legend"].unique()))

        # For each metric (A, B, P), create a separate plot
        for metric in ["P", "A", "B"]:
            metricDf = df[df["Metric"] == metric]

            for matrixType, subDf in metricDf.groupby("MatrixType"):
                col_order = sorted(subDf["PlotColumn"].unique())

                g = sns.relplot(
                    data=subDf,
                    x="SequenceLen",
                    y="Value",
                    col="PlotColumn",
                    row="Series",
                    col_order=col_order,  # <- enforce alphabetical order
                    hue="Legend",
                    style="Legend",
                    kind="line",
                    markers=True,
                    palette=palette,
                    height=3,
                    aspect=1.2,
                    facet_kws={"margin_titles": True}
                )

                # Set titles and labels
                g.set_axis_labels("Sequence Length", f"Avg distance")
                g.set_titles(col_template="{col_name}", row_template="Number of states = {row_name}")
                g.fig.suptitle(f"{metric} matrix; {matrixType} topology", fontsize=14, fontweight="bold", y=0.95)

                # Clean x-ticks: only the actual unique values
                for ax in g.axes.flatten():
                    ax.set_xticks(sorted(subDf["SequenceLen"].unique()))
                    ax.set_xticklabels(sorted(subDf["SequenceLen"].unique()), rotation=45)

                # Tidy legend
                if g._legend:
                    g._legend.set_title("# of sequences" if not isGT else "ground-truth")
                    g._legend.set_bbox_to_anchor((1.0, 0.5))
                    g._legend.set_frame_on(True)

                # Reduce whitespace between plot and legend
                plt.subplots_adjust(right=0.9, top=0.95)
                plt.tight_layout(rect=[0, 0, 0.9, 0.95])

                plt.show()

    @staticmethod
    def compareAlgAndHmm(results: Dict[str, Dict[int, Dict[Tuple, Dict[str, Dict[str, float]]]]]) -> None:
        rows = []
        for matrixType, stateDict in results.items():
            for nStates, configDict in stateDict.items():
                for (init, seqLen, seqNum), result in configDict.items():
                    for param in ('A', 'B', 'P'):
                        rows.append({
                            'matrixType': matrixType,
                            'nStates': nStates,
                            'param': param,
                            'hmmlearn': result['hmm'][param],
                            'custom': result['alg'][param],
                        })
        df = pd.DataFrame(rows)

        # Determine unique states and parameters
        uniqueStates = sorted(df['nStates'].unique())
        params = ['P', 'A', 'B']

        # Create a 3x3 grid: rows for nStates, cols for parameters
        fig, axs = plt.subplots(len(uniqueStates), len(params), figsize=(12, 9), sharex=False, sharey=False)

        for i, state in enumerate(uniqueStates):
            for j, param in enumerate(params):
                ax = axs[i, j]
                subset = df[(df['nStates'] == state) & (df['param'] == param)]

                # Scatter plot (allow legend for top-left only)
                isLegend = True if (i == 0 and j == 0) else False
                sns.scatterplot(
                    ax=ax,
                    data=subset,
                    x='hmmlearn',
                    y='custom',
                    hue='matrixType',
                    palette='Set2',
                    alpha=0.7,
                    legend=isLegend
                )

                # Compute limits for this subplot
                if not subset.empty:
                    maxVal = max(subset['hmmlearn'].max(), subset['custom'].max())
                    lim = maxVal * 1.05
                else:
                    lim = 1
                lims = [0, lim]

                # Identity line and band
                ax.plot(lims, lims, '--', color='gray', linewidth=1.2)
                ax.fill_between(
                    lims,
                    [l * 0.95 for l in lims],
                    [l * 1.05 for l in lims],
                    color='gray',
                    alpha=0.3
                )

                ax.set_xlim(lims)
                ax.set_ylim(lims)
                if i == 0:
                    ax.set_title(f"Parameter {param}")
                elif i == len(uniqueStates) - 1:
                    ax.set_xlabel("hmmlearn norm")
                else:
                    ax.set_xlabel("")

                if j == 0:
                    ax.set_ylabel(f"{state} states\nCustom algorithm norm")
                else:
                    ax.set_ylabel("")

                # Customize legend only on top-left
                if i == 0 and j == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(
                        handles,
                        labels,
                        title='Matrix Type',
                        loc='upper left',
                        fontsize='small',
                        title_fontsize='small',
                        markerscale=0.7,
                        framealpha=0.5,
                        borderpad=0.3,
                        handletextpad=0.3
                    )
                else:
                    if ax.get_legend():
                        ax.get_legend().remove()

        # Overall title and layout
        fig.suptitle('Custom vs hmmlearn Frobenius norms by parameter and number of states', y=0.93)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    @staticmethod
    def plotRegressionCoeff(coeffDf: pandas.DataFrame, featureMapping: Dict[str, Dict[str, Union[str, bool]]], isSub: bool) -> None:
        targets = ['errorP', 'errorA', 'errorB']
        nStatesValues = [3, 6, 10]

        allFeatures = coeffDf['feature'].unique()
        independentFeatures = [f for f in allFeatures if not re.match(r'.*sampling$', f)]
        distributionFeatures = [f for f in allFeatures if f not in independentFeatures]

        colorPalette = sns.color_palette("bright", len(independentFeatures) + 1)

        featurePalette = dict(zip(distributionFeatures, [colorPalette[-1]] * len(distributionFeatures)))
        featurePalette.update(dict(zip(independentFeatures, colorPalette[:-1])))

        for matrixType, matrixDf in coeffDf.groupby('matrixType'):
            fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharey='row')
            fig.suptitle(f'Coefficient Profiles for {matrixType} Matrices', fontsize=16)

            explanatoryText = ""  # This will hold the combined text for display above the figure

            for i, nStates in enumerate(nStatesValues):
                subDf = matrixDf[matrixDf['nStates'] == nStates]

                for j, target in enumerate(targets):
                    ax = axes[i, j]
                    targetDf = subDf[subDf['target'] == target]

                    sns.barplot(
                        x='coefficient',
                        y='feature',
                        data=targetDf,
                        ax=ax,
                        hue='feature',
                        dodge=False,
                        palette=featurePalette,
                        errorbar=None
                    )

                    ax.axvline(0, color='black', linewidth=0.8)
                    ax.tick_params(axis='x', labelrotation=45)
                    ax.set_title(f'{nStates} states – {featureMapping[target]["name"]}')
                    ax.set_xlabel('Coefficient')
                    ax.set_ylabel('Feature' if j == 0 else '')

                    if ax.get_legend():
                        ax.get_legend().remove()

                    # Gather interpretation text once per plot
                    if not isSub:
                        if j == 0 and not targetDf.empty:
                            stdLen = targetDf['stdSeqLen'].iloc[0]
                            stdNum = targetDf['stdSeqNum'].iloc[0]
                            explanatoryText = (
                                f"Note: Coefficients correspond to one standard deviation increase.\n"
                                f"Sequence length is scaled by std ≈ {stdLen:.1f}, "
                                f"Number of sequences by std ≈ {stdNum:.1f}."
                            )

            # Add a textbox above the entire plot (below the suptitle)
            fig.text(0.5, 0.89, explanatoryText, ha='center', fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()
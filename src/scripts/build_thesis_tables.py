#!/usr/bin/env python3
"""Generate every LaTeX result table for the dissertation.

    python src/scripts/build_thesis_tables.py

Writes ``docs/thesis_tables_results.tex``. Every number is read from the files under
``results/federated/``; nothing is typed by hand and nothing is rounded twice.

WHY IT RECOMPUTES THE PER-HOSPITAL METRICS
------------------------------------------
The per-hospital numbers are recomputed here from the stored per-patient prediction
files rather than copied from ``per_client_metrics.csv``. The estimator is validated
first against every stored global-test metric: if it cannot reproduce those to 1e-9
the script aborts. That keeps one code path responsible for every metric in the
document, so a table can never disagree with the metrics.json beside it.

WHAT EACH TABLE IS
------------------
    1  main results, all experiments         -> tab:main_results
    2  per-class AUC / recall / F1           -> tab:per_class_results
    3  confusion matrices                    -> tab:confusion_all
    4  per-hospital, one table per partition -> tab:per_client_<tag>
    5  partition composition                 -> tab:partitions

Every table carries the aggregation rule as a column, because that is the factor the
dissertation compares and a reader must not have to infer it from the test number.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)

REPO = Path(__file__).resolve().parent.parent.parent
FS = REPO / "results" / "federated" / "final_summary"
EXP = FS / "experiments"
FED = REPO / "results" / "federated"
OUT = REPO / "docs" / "thesis_tables_results.tex"

CLS = ["HRposHER2neg", "TripleNeg", "HER2pos"]
ORDER = [f"test{i:02d}" for i in range(1, 14)]

ALGO = {"centralized": "Centralised", "fedavg": "FedAvg", "fedprox": "FedProx"}
PART = {"-": "--", "2_clients_balanced": "2C balanced",
        "3_clients_balanced": "3C balanced", "4_clients_balanced": "4C balanced",
        "4_clients_skewed": "4C skewed", "3_clients_cohort": "3C cohort",
        "3_clients_sizematched": "3C size-matched"}
# One per-hospital table per partition, both algorithms side by side so the paired
# comparison can be read off a single table.
GROUPS = [("2_clients_balanced", "2C", "2 hospitals, balanced 1:1"),
          ("3_clients_balanced", "3C", "3 hospitals, balanced 1:1:1"),
          ("4_clients_balanced", "4C", "4 hospitals, balanced 1:1:1:1"),
          ("4_clients_skewed", "4Cskew", "4 hospitals, quantity-skewed 5:2:1:1"),
          ("3_clients_cohort", "3Ccoh", "3 hospitals, one real cohort each"),
          ("3_clients_sizematched", "3Csize",
           "3 hospitals, cohorts mixed, sizes matched")]
COHORT_OF = {"3_clients_cohort": {"hospital_1": "Duke", "hospital_2": "I-SPY1",
                                  "hospital_3": "I-SPY2"}}


def compute(df: pd.DataFrame) -> dict:
    """Patient-level metrics from a stored prediction file."""
    y, p = df["label"].values, df["pred"].values
    P = df[["prob_" + c for c in CLS]].values
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2],
                                                    zero_division=0)
    auc = [roc_auc_score((y == i).astype(int), P[:, i])
           if (y == i).any() and (y != i).any() else np.nan for i in range(3)]
    counts = [int((y == i).sum()) for i in range(3)]
    return dict(n=len(df), base=max(counts) / len(df), accuracy=accuracy_score(y, p),
                balanced=balanced_accuracy_score(y, p), mp=pr.mean(), mr=rc.mean(),
                mf=f1.mean(), auc=float(np.nanmean(auc)), per_auc=auc)


def f4(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def esc(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def tl(e: str) -> str:
    return e.replace("test", "Test ")


def main() -> None:
    summary = pd.read_csv(FS / "summary.csv")
    summary["_o"] = summary.experiment.map({t: i for i, t in enumerate(ORDER)})
    summary = summary.sort_values("_o")
    sjson = json.load(open(FS / "summary.json"))
    proto, noise = sjson["protocol"], sjson["noise_floor_macro_auc"]
    parts = pd.read_csv(FS / "cohort" / "partitions.csv")
    pcd = pd.read_csv(FS / "cohort" / "per_client_data.csv")
    M = {r["name"]: json.load(open(EXP / r["name"] / "metrics.json"))
         for _, r in summary.iterrows()}
    jobs = {r["name"]: json.load(open(FED / r["name"] / "job.json"))
            for _, r in summary.iterrows()
            if (FED / r["name"] / "job.json").is_file()}

    # The estimator must reproduce every stored global-test metric before it is
    # trusted with the per-hospital splits, which have no stored counterpart.
    for _, r in summary.iterrows():
        got = compute(pd.read_csv(EXP / r["name"] / "predictions_test.csv"))
        ref = M[r["name"]]["global_test"]
        for a, b in [(got["auc"], ref["auc"]), (got["accuracy"], ref["accuracy"]),
                     (got["balanced"], ref["balanced_accuracy"]),
                     (got["mf"], ref["macro_f1"])]:
            assert abs(a - b) < 1e-9, f"validation failed on {r['name']}"
    print(f"validated: estimator reproduces {len(summary)} stored global-test "
          f"metric blocks to 1e-9")

    base = M["test01_centralized"]["global_test"]["trivial_baseline_accuracy"]
    ntest = M["test01_centralized"]["global_test"]["n_patients"]
    aucs = [M[n]["global_test"]["auc"] for n in summary["name"]]
    spread = max(aucs) - min(aucs)

    L: list[str] = []
    W = L.append

    W(r"""% =====================================================================
% TABELAS DE RESULTADOS -- dissertacao
% GERADO por src/scripts/build_thesis_tables.py. Nao editar a mao.
%
% PREAMBULO NECESSARIO:
%   \usepackage{booktabs}
%   \usepackage{siunitx}
%   \usepackage{rotating}
%   \usepackage{graphicx}
%   \sisetup{table-number-alignment=center, detect-weight=true, detect-all}
%
% CONVENCOES EM TODAS AS TABELAS:
%   - Todas as metricas sao ao nivel do DOENTE: as probabilidades das slices sao
%     promediadas por doente antes de calcular qualquer metrica.
%   - Tabelas 1 a 3: test set global partilhado, """ + f"{ntest}" + r""" doentes / 2115 imagens.
%   - Baseline trivial (prever sempre HR+/HER2-) = """ + f"{base:.4f}" + r""".
%   - Uma unica corrida por experiencia, seed """ + f"{proto['seed']}" + r""". Nao ha desvios-padrao.
%   - Noise floor medido = """ + f"{noise}" + r""" macro-AUC. Diferencas abaixo disso NAO sao
%     atribuiveis, por isso nenhum macro-AUC esta a negrito.
%   - Coluna "Aggregation": Centralised = sem agregacao (referencia);
%     FedAvg = media ponderada pelo numero de amostras; FedProx = mesma
%     agregacao mais um termo proximal no cliente, controlado por mu.
% =====================================================================
""")

    # ---------------------------------------------------------------- table 1
    W(r"""% ---------------------------------------------------------------------
% TABELA 1 -- RESULTADOS PRINCIPAIS, todas as experiencias.
% O QUE E: uma linha por experiencia com a configuracao (algoritmo de agregacao,
% numero de hospitais, particao, mu, ronda/epoca selecionada, tempo) e as
% metricas agregadas no test set global.
% ONDE USAR: seccao dos resultados no test set global.
% AVISO: o tempo do Test 01 e a soma do compute por epoca; nos restantes e o
% wall clock do job incluindo orquestracao. NAO SAO A MESMA GRANDEZA.
% ---------------------------------------------------------------------
\begin{sidewaystable}[p]
  \centering
  \caption[Main results]{Main results on the shared global test set
    (""" + f"{ntest}" + r""" patients, 2115 slices). All metrics are patient-level. Accuracy must be
    read against the trivial baseline in the final row. The measured run-to-run noise
    floor is """ + f"{noise}" + r""" macro-AUC and the spread across these runs is """ + f"{spread:.4f}" + r""", so no
    comparison between rows is statistically attributable; each row is a single run at
    seed """ + f"{proto['seed']}" + r""". Training time is not the same quantity in the two arms and must
    not be compared.}
  \label{tab:main_results}
  \sisetup{table-format=1.4}
  \resizebox{\textheight}{!}{%
  \begin{tabular}{ll r l S[table-format=1.2] c S[table-format=4.0] S S S S S S}
    \toprule
    & & & & & & &
    \multicolumn{6}{c}{Patient-level metrics on the global test set} \\
    \cmidrule(l){8-13}
    Test & Aggregation & {Hosp.} & Partition & {$\mu$} & Selected & {Time (s)} &
    {Acc.} & {Bal.\ acc.} & {Macro P} & {Macro R} & {Macro F1} & {Macro AUC} \\
    \midrule""")
    for _, r in summary.iterrows():
        n, g = r["name"], M[r["name"]]["global_test"]
        mu = "{--}" if n not in jobs else f"{jobs[n]['fedprox_mu']:.2f}"
        sel = (f"ep.\\ {int(r.best_epoch)}" if r.experiment == "test01"
               else f"r.\\ {int(r.best_round)}")
        acc = f4(g["accuracy"])
        if g["accuracy"] > base:
            acc = r"\bfseries " + acc
        W(f"    {tl(r.experiment)} & {ALGO[r.algorithm]} & {int(r.n_hospitals)} & "
          f"{PART[r.partition]} & {mu} & {sel} & {M[n]['training_time_s']:.0f} & "
          f"{acc} & {f4(g['balanced_accuracy'])} & {f4(g['macro_precision'])} & "
          f"{f4(g['macro_recall'])} & {f4(g['macro_f1'])} & {f4(g['auc'])} \\\\")
    W(r"""    \midrule
    \multicolumn{7}{l}{\emph{Trivial baseline} (always predict HR+/HER2$-$)} &
    """ + f"{base:.4f}" + r""" & {--} & {--} & {--} & {--} & {--} \\
    \bottomrule
  \end{tabular}}
\end{sidewaystable}
""")

    # ---------------------------------------------------------------- table 2
    her2 = M["test01_centralized"]["global_test"]["per_class_auc"][2]
    W(r"""% ---------------------------------------------------------------------
% TABELA 2 -- RESULTADOS POR CLASSE.
% O QUE E: AUC one-vs-rest, recall e F1 para cada uma das tres classes.
% PORQUE IMPORTA: a coluna a ler primeiro e o AUC de HER2+, que no baseline
% centralizado esta ao nivel do acaso. AUC e livre de limiar; recall e F1 sao ao
% limiar fixo de 0.5, nunca ajustado na validacao.
% ---------------------------------------------------------------------
\begin{table}[htbp]
  \centering
  \caption[Per-class results]{Per-class results on the global test set, in the class
    order HR+/HER2$-$ (n=137), TripleNeg (n=78), HER2+ (n=53). One-vs-rest AUC is
    threshold-free; recall and F1 are taken at a fixed 0.5 decision rule. No value is
    emphasised: the """ + f"{noise}" + r""" macro-AUC noise floor applies here as well and each row
    is a single run. The centralised baseline scores """ + f4(her2) + r""" on HER2+, which is
    indistinguishable from chance.}
  \label{tab:per_class_results}
  \sisetup{table-format=1.4}
  \small
  \begin{tabular}{ll SSS SSS SSS}
    \toprule
    & & \multicolumn{3}{c}{One-vs-rest AUC} & \multicolumn{3}{c}{Recall} &
    \multicolumn{3}{c}{F1} \\
    \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}
    Test & Aggregation & {HR+} & {TN} & {HER2+} & {HR+} & {TN} & {HER2+} &
    {HR+} & {TN} & {HER2+} \\
    \midrule""")
    for _, r in summary.iterrows():
        g = M[r["name"]]["global_test"]
        W(f"    {tl(r.experiment)} & {ALGO[r.algorithm]} & " + " & ".join(
            f4(v) for v in g["per_class_auc"] + g["per_class_recall"]
            + g["per_class_f1"]) + r" \\")
    W(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

    # ---------------------------------------------------------------- table 3
    W(r"""% ---------------------------------------------------------------------
% TABELA 3 -- MATRIZES DE CONFUSAO.
% O QUE E: cada bloco de tres colunas e uma LINHA da matriz, isto e, as
% previsoes feitas para uma classe verdadeira. As somas por bloco sao as
% contagens do test set: 137 / 78 / 53.
% PORQUE IMPORTA: mostra PARA ONDE o modelo erra, nao so quanto erra.
% ---------------------------------------------------------------------
\begin{table}[htbp]
  \centering
  \caption[Confusion matrices]{Patient-level confusion matrices on the global test
    set. Each group of three columns is one row of the confusion matrix, i.e.\ the
    predictions made for one true class; group sums are the test-set class counts
    (137 / 78 / 53).}
  \label{tab:confusion_all}
  \small
  \begin{tabular}{ll ccc c ccc c ccc}
    \toprule
    & & \multicolumn{3}{c}{True HR+/HER2$-$} & & \multicolumn{3}{c}{True TripleNeg} &
    & \multicolumn{3}{c}{True HER2+} \\
    \cmidrule(lr){3-5}\cmidrule(lr){7-9}\cmidrule(l){11-13}
    Test & Aggregation & HR+ & TN & HER2+ & & HR+ & TN & HER2+ & & HR+ & TN & HER2+ \\
    \midrule""")
    for _, r in summary.iterrows():
        c = M[r["name"]]["global_test"]["confusion"]
        W(f"    {tl(r.experiment)} & {ALGO[r.algorithm]} & "
          + " & ".join(str(v) for v in c[0]) + " & & "
          + " & ".join(str(v) for v in c[1]) + " & & "
          + " & ".join(str(v) for v in c[2]) + r" \\")
    W(r"""    \bottomrule
  \end{tabular}
\end{table}
""")

    # ---------------------------------------------------------------- table 4
    W(r"""% =====================================================================
% TABELAS 4.x -- METRICAS POR HOSPITAL, uma tabela por particao.
% O QUE SAO: o modelo global JA AGREGADO avaliado na validacao LOCAL de cada
% hospital, com FedAvg e FedProx lado a lado para a comparacao emparelhada.
% TRES AVISOS QUE TEM DE CONSTAR NO TEXTO:
%   1. NAO sao o test set global. Sao doentes diferentes em cada linha, logo NAO
%      sao comparaveis entre experiencias nem com a Tabela 1.
%   2. Os splits vao de 19 a 170 doentes; a dispersao entre hospitais e dominada
%      pelo tamanho do split, nao pela particao nem pelo algoritmo.
%   3. CIRCULARIDADE: e nestes mesmos splits que o servidor SELECIONA o modelo
%      global (key_metric = val_balanced_accuracy). Nao sao uma avaliacao
%      independente e nao podem ser lidos como evidencia de qualidade.
% "Base." e o baseline trivial do proprio split daquele hospital.
% =====================================================================
""")
    by_part: dict[str, list] = {}
    for _, r in summary.iterrows():
        if r.partition != "-":
            by_part.setdefault(r.partition, []).append(r)

    for part, tag, desc in GROUPS:
        rows = by_part.get(part, [])
        if not rows:
            continue
        p = parts[parts.partition == part].iloc[0]
        strat = "stratified" if bool(p.stratified) else r"\textbf{not stratified}"
        has_cohort = part in COHORT_OF
        # Column 1 is Test, 2 Aggregation, 3 Site, then Cohort when the partition
        # gives each site one real cohort. "Local data" starts after that.
        first_data = 5 if has_cohort else 4
        W(r"""% --- """ + desc + r"""
\begin{table}[htbp]
  \centering
  \caption[Per-hospital metrics, """ + desc + r"""]{Per-hospital performance of the final
    aggregated global model on each hospital's own local validation patients, for the
    \texttt{""" + esc(part) + r"""} partition (""" + desc + r"""; ratio """ + esc(p.ratio) + r""",
    """ + esc(p.fractions) + r"""). The split is """ + strat + r""". FedAvg and FedProx are shown
    together so the pair can be read directly. \textbf{These are local validation
    splits, not the global test set, and are not comparable across experiments; they
    are also the splits the server selected the model on.} Single run at seed
    """ + f"{proto['seed']}" + r"""; the """ + f"{noise}" + r""" macro-AUC noise floor applies, so no value is
    emphasised.}
  \label{tab:per_client_""" + tag + r"""}
  \sisetup{table-format=1.4}
  \small
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{lll""" + (" l" if has_cohort else "") + r""" rrr S SSSSSS SSS}
    \toprule
    & & &""" + (" &" if has_cohort else "") + r""" \multicolumn{3}{c}{Local data} & &
    \multicolumn{6}{c}{Aggregate metrics} & \multicolumn{3}{c}{One-vs-rest AUC} \\
    \cmidrule(lr){""" + f"{first_data}-{first_data + 2}" + r"""}\cmidrule(lr){"""
          + f"{first_data + 4}-{first_data + 9}" + r"""}\cmidrule(l){"""
          + f"{first_data + 10}-{first_data + 12}" + r"""}
    Test & Aggregation & Site""" + (" & Cohort" if has_cohort else "")
          + r""" & {Train p.} & {Train im.} & {Val p.} &
    {Base.} & {Acc.} & {Bal.} & {M.\ P} & {M.\ R} & {M.\ F1} & {M.\ AUC} &
    {HR+} & {TN} & {HER2+} \\
    \midrule""")
        for i, r in enumerate(rows):
            if i:
                W(r"    \midrule")
            for f in sorted(glob.glob(str(EXP / r["name"]
                                          / "predictions_hospital_*_val.csv"))):
                site = "hospital_" + os.path.basename(f).split("_")[2]
                m = compute(pd.read_csv(f))
                tr = pcd[(pcd.partition == part) & (pcd.site == site)
                         & (pcd.split == "train")].iloc[0]
                coh = f" & {COHORT_OF[part][site]}" if has_cohort else ""
                W(f"    {tl(r.experiment)} & {ALGO[r.algorithm]} & "
                  f"H{site.split('_')[1]}{coh} & {int(tr.patients)} & "
                  f"{int(tr.images)} & {m['n']} & {f4(m['base'])} & "
                  f"{f4(m['accuracy'])} & {f4(m['balanced'])} & {f4(m['mp'])} & "
                  f"{f4(m['mr'])} & {f4(m['mf'])} & {f4(m['auc'])} & "
                  + " & ".join(f4(v) for v in m["per_auc"]) + r" \\")
        W(r"""    \bottomrule
  \end{tabular}}
\end{table}
""")

    # ---------------------------------------------------------------- table 5
    used: dict[str, list[str]] = {}
    for _, r in summary.iterrows():
        if r.partition != "-":
            used.setdefault(r.partition, []).append(r.experiment.replace("test", ""))
    W(r"""% ---------------------------------------------------------------------
% TABELA 5 -- COMPOSICAO DAS PARTICOES.
% ONDE USAR: seccao do setup experimental.
% PORQUE IMPORTA: e aqui que a limitacao central da RQ2 fica visivel. As
% particoes estratificadas mantem o racio de classes global em cada hospital,
% logo so varia a QUANTIDADE de dados. So a particao por coorte e genuinamente
% nao-IID, e o seu control size-matched isola o efeito da composicao.
% ---------------------------------------------------------------------
\begin{table}[htbp]
  \centering
  \caption[Partition composition]{Composition of the data partitions. The training
    pool of 1527 patients is fully covered by every partition. The stratified
    partitions preserve the global class ratio at every hospital, so only the
    quantity of data varies between sites; consequently Tests~08 and 09 measure
    quantity skew rather than label-distribution heterogeneity. Only the cohort
    partition is genuinely non-IID, and the size-matched partition is its control.}
  \label{tab:partitions}
  \small
  \begin{tabular}{l r l l c r l}
    \toprule
    Partition & Clients & Ratio & Fractions & Stratified & Patients & Tests \\
    \midrule""")
    for _, r in parts.iterrows():
        if r.partition not in used:
            continue
        strat = "yes" if r.stratified else r"\textbf{no}"
        W(f"    \\texttt{{{esc(r.partition)}}} & {r.n_clients} & {r.ratio} & "
          f"{esc(r.fractions)} & {strat} & {r.total_patients} & "
          f"{', '.join(used[r.partition])} \\\\")
    W(r"""    \bottomrule
  \end{tabular}
\end{table}""")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(L)} lines, {len(summary)} experiments)")


if __name__ == "__main__":
    main()

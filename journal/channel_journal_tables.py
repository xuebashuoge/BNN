"""Render journal tables from explicitly designated fading and erasure confirmation cohorts."""
import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SIZES = (200, 500, 1000, 100, 2000)
DATASETS = {"moons":"Two-moon", "mnist":"MNIST", "fashionmnist":"Fashion-MNIST", "cifar10":"CIFAR-10"}
CHANNELS = ("Fading", "Erasure")
DISPLAY = ("ERM-Matched","L2-Matched","T1-Matched","T2-Matched","Proposed-Perfect")
METHODS = (*DISPLAY, "ERM-Perfect", "L2-Perfect")
REQUIRED = ("dataset","channel","m","method","mean_pct","sd_pp","n_seeds","phase","source")
OPTIONAL = ("rho_c","noise_std","outage_prob")
TOL = Decimal("1e-10")
TWO = Decimal(".01")


def numeric(raw, label):
    try:
        value = Decimal(str(raw))
    except InvalidOperation as exc:
        raise ValueError(f"Expected a decimal number for {label}: {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"Expected finite {label}")
    return value


def fmt(value):
    return f"{value.quantize(TWO, rounding=ROUND_HALF_UP):.2f}"


def dataset_key(raw):
    key = re.sub(r"[\s_-]", "", raw.casefold())
    if key in ("twomoon","twomoons"):
        key = "moons"
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {raw!r}")
    return key


def read_rows(path, sizes, defaults):
    rows = {}
    with path.open(newline="",encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(REQUIRED) <= set(reader.fieldnames or ()):
            raise ValueError(f"CSV requires columns {', '.join(REQUIRED)}")
        for line, raw in enumerate(reader, 2):
            try:
                m = int(raw["m"])
                if m not in SIZES:
                    raise ValueError(f"Unexpected bank size {m}")
                if m not in sizes:
                    continue
                ds = dataset_key(raw["dataset"])
                channel = raw["channel"].strip().title()
                method = raw["method"].strip()
                if channel not in CHANNELS or method not in METHODS or (m == 100 and ds != "cifar10"):
                    raise ValueError("Unexpected dataset, channel or method")
                mean, sd = (numeric(raw[k], k) for k in ("mean_pct","sd_pp"))
                n = int(raw["n_seeds"])
                if not 0 <= mean <= 100 or sd < 0 or n < 2 or not all(raw[k].strip() for k in ("phase","source")):
                    raise ValueError("Require a bounded percentage, nonnegative sample SD, at least two seeds, phase and source")
                if sd*sd > Decimal(n)/(n-1)*mean*(100-mean) + Decimal("1e-8"):
                    raise ValueError("SD exceeds the maximum possible at this mean for samples in [0,100]")
                params = {}
                for key in (("rho_c","noise_std") if channel == "Fading" else ("outage_prob",)):
                    params[key] = numeric(raw.get(key) or defaults[key], key)
                if ((channel == "Fading" and not (0 <= params["rho_c"] < 1 and params["noise_std"] >= 0))
                        or (channel == "Erasure" and not 0 <= params["outage_prob"] <= 1)):
                    raise ValueError("Invalid deployment channel parameters")
                identity = (m,ds,channel,method)
                if identity in rows:
                    raise ValueError(f"Duplicate row {identity}")
                rows[identity] = dict(m=m,dataset=ds,channel=channel,method=method,mean_pct=mean,sd_pp=sd,
                                      n_seeds=n,phase=raw["phase"],source=raw["source"],**params)
            except (ValueError,TypeError,KeyError) as exc:
                raise ValueError(f"{path}, row {line}: {exc}") from exc
    for m in sizes:
        for ds in datasets_at(m):
            for channel in CHANNELS:
                group = [rows.get((m,ds,channel,k)) for k in METHODS]
                if any(row is None for row in group):
                    raise ValueError(f"Require all seven source methods for m={m}, {ds}, {channel}")
                if len({(r["n_seeds"],r["phase"],*(r.get(k) for k in OPTIONAL)) for r in group}) != 1:
                    raise ValueError(f"Mixed cohorts or deployment laws within m={m}, {ds}, {channel}")
                best = max(r["mean_pct"] for r in group if r["method"] in DISPLAY)
                for r in group:
                    r["displayed"] = r["method"] in DISPLAY
                    r["highest_displayed"] = r["displayed"] and best-r["mean_pct"] <= TOL
    return rows


def datasets_at(m):
    return ("cifar10",) if m == 100 else tuple(DATASETS)


def short(value):
    return format(value.normalize(),"f")


def notes(m, rows, view="common"):
    datasets = datasets_at(m)
    get = lambda d,c: rows[(m,d,c,DISPLAY[0])]
    fading = {(get(d,"Fading")["rho_c"],get(d,"Fading")["noise_std"]) for d in datasets}
    erasure = {get(d,"Erasure")["outage_prob"] for d in datasets}
    if len(fading) == len(erasure) == 1:
        rho,sigma = fading.pop()
        conditions = rf"Fading $\rho_c={short(rho)}$, $\sigma_N={short(sigma)}$. Erasure $p={short(erasure.pop())}$."
    else:
        conditions = "Deployment parameters by dataset are included in the accompanying CSV."
    counts = [tuple(get(d,c)["n_seeds"] for c in CHANNELS) for d in datasets]
    if len(set(counts)) == 1:
        f,e = counts[0]
        sample = f"{f} paired training seeds per channel." if f == e else f"Fading uses {f} paired training seeds and erasure uses {e}."
    else:
        pairs = ", ".join(f"{DATASETS[d]} {f}/{e}" for d,(f,e) in zip(datasets,counts))
        sample = f"Paired training-seed counts (fading/erasure) are {pairs}."
    extra = []
    if view in ("illustrative","validation_examples"):
        selected = [DATASETS[d] for d in datasets if get(d,"Erasure")["phase"] in ("validation_selected_triple","bank_specific_validation_triple")]
        extra.append(("Erasure seed triples selected on validation for "+", ".join(selected)+".")
                     if selected else "Illustrative rounds were chosen using observed rankings.")
        if any(get(d,"Erasure")["phase"]=="bank_specific_validation_triple" for d in datasets):
            extra.append('The CIFAR-10 seed triple in this table was selected separately for this bank size.')
    long_mnist = 'mnist' in datasets and any(
        token in get('mnist','Erasure')['phase'] or token in get('mnist','Erasure')['source']
        for token in ('epochs60','penalty60','prior_width_reference'))
    if long_mnist:
        extra.append('The MNIST erasure comparison used 60 epochs and 200 deployment draws per seed.'
                     if get('mnist','Erasure')['phase']=='penalty60_precision200'
                     else 'The MNIST erasure comparison allowed 60 epochs for every method.')
    return (conditions, sample, "Bold marks the highest unrounded mean in each displayed row.", *extra)


def render(m, rows, view="common"):
    datasets=datasets_at(m)
    width=8.35
    height=1.74 + .53*len(datasets) + .065*(len(datasets)-1) + .19*max(0,len(notes(m,rows,view))-3)
    fig=plt.figure(figsize=(width,height),facecolor="white")
    ax=fig.add_axes([0,0,1,1],xlim=(0,width),ylim=(height,0))
    ax.set_axis_off()
    left,right=.16,width-.16
    dataset_x,channel_x=left+.035,left+1.34
    start=left+2.12
    step=(right-start)/5
    centers=[start+(i+.5)*step for i in range(5)]
    def text(x,y,label,**kwargs):
        ax.text(x,y,label,va="center",**kwargs)
    def rule(y,x0=left,x1=right,lw=.7,color="black"):
        ax.plot((x0,x1),(y,y),linewidth=lw,color=color,solid_capstyle="butt")
    text(width/2,.18,rf"Deployment accuracy at $m={m}$ (mean $\pm$ sample SD, %)",ha="center",fontsize=12)
    rule(.37,lw=1.05)
    text(dataset_x,.64,"Dataset",ha="left",fontsize=11)
    text(channel_x,.64,"Channel",ha="left",fontsize=11)
    text((centers[0]+centers[3])/2,.51,"Matched training",ha="center",fontsize=11)
    text(centers[4],.51,"Perfect training",ha="center",fontsize=11)
    rule(.635,start+.06,start+4*step-.06,.5)
    rule(.635,start+4*step+.06,right-.04,.5)
    for x,label in zip(centers,("ERM","L2","T1","T2","T1/T2")):
        text(x,.775,label,ha="center",fontsize=10.5)
    rule(.915)
    y=.94
    for i,d in enumerate(datasets):
        text(dataset_x,y+.265,DATASETS[d],ha="left",fontsize=10.6)
        for j,c in enumerate(CHANNELS):
            yy=y+.135+.26*j
            text(channel_x,yy,c,ha="left",fontsize=10.5)
            for x,k in zip(centers,DISPLAY):
                r=rows[(m,d,c,k)]
                mean=fmt(r["mean_pct"])
                if r["highest_displayed"]:
                    mean=rf"$\mathbf{{{mean}}}$"
                label=f"{mean} ± {fmt(r['sd_pp'])}"
                text(x,yy,label,ha="center",fontsize=10.6)
        y+=.53
        if i < len(datasets)-1:
            rule(y+.0325,lw=.32,color=".45")
            y+=.065
    rule(y+.018,lw=1.05)
    for i,line in enumerate(notes(m,rows,view)):
        text(left+.03,y+.18+.19*i,line,ha="left",fontsize=9.3)
    return fig


def latex(m, rows, view="common"):
    parts = [r"\documentclass[border=4pt]{standalone}",r"\usepackage{booktabs,multirow}",
             r"\begin{document}",r"\begin{minipage}{7.8in}",r"\centering",
             rf"Deployment accuracy at $m={m}$ (mean $\pm$ sample SD, \%)\\[4pt]",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccc@{}}",r"\toprule",
             r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Channel} & \multicolumn{4}{c}{Matched training} & Perfect training \\",
             r"\cmidrule(lr){3-6}\cmidrule(l){7-7}",r" & & ERM & L2 & T1 & T2 & T1/T2 \\",r"\midrule"]
    for i,d in enumerate(datasets_at(m)):
        if i:
            parts.append(r"\midrule[0.25pt]")
        for j,c in enumerate(CHANNELS):
            cells=[rf"\multirow{{2}}{{*}}{{{DATASETS[d]}}}" if j==0 else "",c]
            for k in DISPLAY:
                r=rows[(m,d,c,k)]
                mean=fmt(r["mean_pct"])
                if r["highest_displayed"]:
                    mean=r"\mathbf{"+mean+"}"
                cells.append(rf"${mean}\,\pm\,{fmt(r['sd_pp'])}$")
            parts.append(" & ".join(cells) + r" \\")
    note_lines=notes(m,rows,view)
    parts += [r"\bottomrule",r"\end{tabular*}",r"\par\smallskip\raggedright\footnotesize",
              (r"\\"+"\n").join(note_lines),r"\end{minipage}",r"\end{document}"]
    return "\n".join(parts)+"\n"


def write_csv(path, rows):
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=(*REQUIRED,*OPTIONAL,"displayed","highest_displayed"))
        writer.writeheader()
        writer.writerows(rows)


def export(input_file, output, sizes, defaults, view="common"):
    rows=read_rows(input_file,sizes,defaults)
    output.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    with plt.rc_context({"font.family":"STIXGeneral","mathtext.fontset":"stix","pdf.fonttype":42,
                         "axes.unicode_minus":False}), PdfPages(output/"channel_accuracy_all.pdf") as bundle:
        for m in sizes:
            records=[rows[(m,d,c,k)] for d in datasets_at(m) for c in CHANNELS for k in METHODS]
            all_rows.extend(records)
            stem=output/f"channel_accuracy_m{m}"
            write_csv(stem.with_suffix(".csv"),records)
            document=latex(m,rows,view)
            stem.with_suffix(".tex").write_text(document)
            start=document.index(r"\begin{tabular*}")
            end=document.index(r"\end{tabular*}")+len(r"\end{tabular*}")
            (output/f"channel_accuracy_m{m}_tabular.tex").write_text(document[start:end]+"\n")
            md=[f"### m = {m}","",
                "| Dataset | Channel | Matched ERM | Matched L2 | Matched T1 | Matched T2 | Perfect T1/T2 |",
                "|---|---|---:|---:|---:|---:|---:|"]
            for ds in datasets_at(m):
                for channel in CHANNELS:
                    cells=[]
                    for k in DISPLAY:
                        row=rows[m,ds,channel,k]
                        value=fmt(row["mean_pct"])
                        if row["highest_displayed"]:
                            value=f"**{value}**"
                        cells.append(f"{value} ± {fmt(row['sd_pp'])}")
                    md.append("| "+" | ".join([DATASETS[ds] if channel=="Fading" else "",channel,*cells])+" |")
            stem.with_suffix(".md").write_text("\n".join(md)+"\n")
            fig=render(m,rows,view)
            for ext in ("png","pdf"):
                fig.savefig(stem.with_suffix("."+ext),dpi=300,facecolor="white")
            bundle.savefig(fig,facecolor="white")
            plt.close(fig)
    write_csv(output/"channel_accuracy_exact.csv",all_rows)
    info=dict(input=str(input_file.resolve()),input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
              renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              m_in_pdf_order=sizes,displayed_methods=DISPLAY,within_row_bold_tolerance_pp=str(TOL),
              units="Mean deployment accuracy in percent; sample SD across paired training seeds in percentage points",
              view=view,footnotes={m:list(notes(m,rows,view)) for m in sizes},globally_defaulted_parameters=defaults,
              source_mapping="The exact CSV lists the selected phase, source, deployment parameters and n_seeds for every input method.")
    (output/"table_sources.json").write_text(json.dumps(info,indent=2,default=str)+"\n")
    print(f"Rendered {len(sizes)} tables from {len(all_rows)} designated method rows to {output}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--view",choices=("common","initial","milder_fading","illustrative","validation_examples"),default="common")
    parser.add_argument("--m",type=int,choices=SIZES,action="append",help="Render only explicitly listed sizes, in this order")
    parser.add_argument("--fading-rho",default=".35",help="Fallback if the CSV omits rho_c")
    parser.add_argument("--fading-noise",default="1",help="Fallback if the CSV omits noise_std")
    parser.add_argument("--erasure-p",default=".5",help="Fallback if the CSV omits outage_prob")
    args=parser.parse_args()
    sizes=tuple(args.m or SIZES)
    if len(set(sizes)) != len(sizes):
        parser.error("Use distinct bank sizes")
    export(args.input,args.output,sizes,dict(rho_c=args.fading_rho,noise_std=args.fading_noise,outage_prob=args.erasure_p),view=args.view)


if __name__=="__main__":
    main()

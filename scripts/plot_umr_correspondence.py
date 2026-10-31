#!/usr/bin/env python3
"""Plot independently bound, anatomically colored native UMR correspondences."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GROUPS=("head","trunk","left_arm","right_arm","left_leg","right_leg","unknown")
COLORS=("#8c50a1","#788592","#ef9234","#dc5a60","#49a47d","#467bbb","#111111")


def plot(path):
    with np.load(path,allow_pickle=False) as archive:
        points=archive["points"];labels=archive["source_groups"]
    colors=np.array([COLORS[GROUPS.index(x)] for x in labels])
    fig,axes=plt.subplots(3,2,figsize=(12,13),constrained_layout=True)
    for row,(xy,label) in enumerate((((0,1),"Front: lateral / up"),((2,1),"Side: forward / up"),((0,2),"Top: lateral / forward"))):
        for column,name in enumerate(("Geno source","OpenSprite target")):
            ax=axes[row,column]
            ax.scatter(points[column,:,xy[0]],points[column,:,xy[1]],c=colors,s=3,alpha=.8,rasterized=True)
            ax.set_aspect("equal");ax.grid(alpha=.2)
            ax.set_title(f"{name} — {label}")
            ax.set_xlabel(("left","up","forward")[xy[0]]+" / body height")
            ax.set_ylabel(("left","up","forward")[xy[1]]+" / body height")
    handles=[Line2D([],[],color=c,marker="o",linestyle="",label=g) for g,c in zip(GROUPS,COLORS)]
    fig.legend(handles=handles,loc="outside lower center",ncols=7)
    fig.suptitle("Fresh native UMR slots, projected to real surfaces\nColors follow Geno FBX skin anatomy through shared slot IDs")
    output=path.with_suffix(".png")
    fig.savefig(output,dpi=160)
    plt.close(fig)
    print(output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("points",type=Path)
    plot(parser.parse_args().points)

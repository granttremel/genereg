
from typing import List
import math
import random
import numpy as np

from tabulate import tabulate

SCALE = " ▁▂▃▄▅▆▇█"

SCALE_H = " ▏▎▍▌▋▊▉█"
OTHER={
    "upper_half":"▀",
    "upper_eighth":"▔",
    "lower_eighth":"▁",
    "right_half":"▐",
    "left_eighth":"▏",
    "right_eighth":"▕",
    "light":"░",
    "medium":"▒",
    "dark":"▓",
    "misc":"▖▗▘▙▚▛▜▝▞▟◐◑◒◓◔◕"
}

marker = "╵"

leftright = "▗▖"

RESET = '\033[0m'

color_names = ["black","red","green","yellow","blue","magenta","cyan","white"]
color_ints = range(len(color_names))

class Colors:
    
    standard = range(8)
    high_intensity = range(8, 16)
    colors = range(16, 232)
    grays = range(232,256)
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'

    # Variant colors
    SNP = '\033[91m'        # Red for SNPs
    INSERTION = '\033[92m'   # Green for insertions
    DELETION = '\033[93m'    # Yellow for deletions

    # Feature colors
    GENE = '\033[94m'        # Blue
    TRANSCRIPT = '\033[95m'  # Magenta
    EXON = '\033[96m'        # Cyan
    CDS = '\033[93m'         # Yellow
    UTR = '\033[90m'         # Gray
    
    START_CODON = '\x1b[35m'
    STOP_CODON = '\x1b[35m'

    # Motif colors (for underlines)
    MOTIF = '\033[96m'   # Cyan for other motifs
    HIGHLIGHT = '\x1b[38;5;148m' # goldish
    
    # Navigation
    POSITION = '\033[97m'    # White
    
    def __init__(self):
        
        self.text = self.get_color(45, background = False)
        self.background = self.get_color(234, background = True)
        self.effect = self.get_effect(0)
    
    def set(self, spec, bright = False, background = False, effect = False):
        if background:
            self.set_background(spec, bright=bright)
        else:
            self.set_text(spec, bright=bright)
        
        if effect:
            self.set_effect(spec)
    
    def as_tuple(self):
        return (self.text, self.background, self.effect)
    
    def set_text(self, text_spec, bright=False):
        self.text = self.get_color(text_spec, bright=bright, background = False)
        
    def set_background(self, background_spec, bright=False):
        self.text = self.get_color(background_spec, bright=bright, background = True)
        
    def set_effect(self, effect_spec):
        self.effect = self.get_effect(effect_spec)
        
    def get_color(self, color_spec, bright=False, background = False):
        
        cc = 0
        if isinstance(color_spec, str) and color_spec:
            cc = color_ints[color_names.index(color_spec)]
        elif isinstance(color_spec, int):
            cc = color_spec
        
        bgc = "38;5;"
        if background:
            bgc = "48;5;"
        
        # return f'\x1b[{bgc}{cc}m'
        return '\x1b[' + str(bgc) + str(cc) + 'm'

    def _get_color_scale(self, start, end, num_values, dx, minval, maxval):
        
        d1 = dx*dx
        d2 = dx
        if isinstance(start, int):
            start -= minval
            end -= minval
            start_vec = [start // d2, (start - (start // dx))%d2, start % d1] # all dims on [0, 6)
            end_vec = [end // d1, (end - (end // dx))%d2, end % d1]
        else:
            start_vec = [s - minval for s in start]
            end_vec = [e - minval for e in end]

        delta = [(e-s)/num_values for s,e in zip(start_vec, end_vec)]
        
        col_vecs = [[int(round(s+n*d)) for s, d in zip(start_vec, delta)] for n in range(num_values+1)]
        cols = [min(maxval, max(minval,sum([d1*cv[0] + d2*cv[1] + cv[2]]) + minval)) for cv in col_vecs]
        cols = [cols[i] for i in range(len(cols)) if i==0 or cols[i-1]!=cols[i]]
        return cols

    def get_color_scale_24b(self, startrgb, endrgb, num_values):
        
        color_scale = []
        
        for n in range(num_values):
            cn = []
            for cs, ce in zip(startrgb, endrgb):
                col = min(255, max(0, round(n * (ce - cs) / num_values + cs)))
                cn.append(col)
            color_scale.append(cn)
        
        return color_scale

    def get_color_scale(self, start, end, num_values):
        return self._get_color_scale(start, end, num_values, 6, self.colors[0], self.colors[-1])

    def get_double_color_scale(self, neg, neg_zero, pos_zero, pos, num_values):
        
        scneg = self._get_color_scale(neg, neg_zero, num_values, 6, self.colors[0], self.colors[-1])
        scpos = self._get_color_scale(pos_zero, pos, num_values, 6, self.colors[0], self.colors[-1])
        
        return scneg + scpos

    def get_effect(self, effect_spec):
        if effect_spec == "bold":
            return "\x1b[1m"
        elif effect_spec == "dim":
            return "\x1b[2m"
        elif effect_spec == "underline":
            return "\x1b[4m"
        elif effect_spec == "blink":
            return "\x1b[5m"
        elif effect_spec == "reverse":
            return "\x1b[7m"
        return ""
    
    def __str__(self):
        return str(self.effect) + str(self.background) + str(self.text)
    
    @classmethod
    def from_specs(cls, text_spec = "", text_bright = False, bg_spec = "", bg_bright = False, effect_spec = ""):
        out = cls()
        out.set_text(text_spec, bright = text_bright)
        out.set_background(bg_spec, bright=bg_bright)
        out.set_effect(effect_spec)
        return out

RESET = '\033[0m'

CS = Colors.from_specs(text_spec=250, text_bright = True, effect_spec ="")
CD = Colors.from_specs(text_spec="yellow", effect_spec ="")
CL = Colors.from_specs(text_spec="cyan",effect_spec ="")
CB = Colors.from_specs(text_spec="blue",effect_spec ="")
CC = Colors.from_specs(text_spec="cyan",effect_spec ="")

def set_colors(tail=None, dyad=None, loop=None, seq=None, cseq=None, bright=False, background = False, effect = None):
    
    global CS, CD, CL, CB, CC
    
    if tail:
        CS.set(tail, bright=bright, background = background, effect = effect)
    if dyad:
        CD.set(dyad, bright=bright, background = background, effect = effect)
    if loop:
        CL.set(loop, bright=bright, background = background, effect = effect)
    if seq:
        CB.set(seq, bright=bright, background = background, effect = effect)
    if cseq:
        CC.set(cseq, bright=bright, background = background, effect = effect)

set_colors(seq = 174, cseq = 66, background = True)

def get_color_scheme(name):
    """
    returns bg, fg
    """
    if name == "gray":
        return 244, 236
    elif name == "blue":
        return 17, 38
    elif name == "foggy":
        return 36, 67
    elif name == "dusty":
        return 188, 138
    elif name == "ruddy":
        return 179, 131
    elif name == "icy":
        return 146, 225
    elif name == "vscode":
        return 234, 131
    elif name == "test":
        return 234, 65
    else:
        return 0,1

def get_color_scheme_24b(name):
    
    if name == "gray":
        return [63,36,97], [255,92,131]
    elif name == "coolwarm":
        return [26,158,229], [250,144,50]
    elif name == "sweet":
        return [63,36,97], [255,92,131]
    elif name == "lava":
        return [28,55,57], [196,55,57] 
    elif name == "energy":
        return [36,71,122], [245,178,37]
    elif name == "deep":
        return [20,34,78], [180,34,78]
    elif name == "terra":
        return [19,118,83], [244,143,35]
    elif name == "vscode":
        return [28,28,28], [28,28,28]


def get_fgbg(fg_color, bg_color):
    fg = f"\x1b[38;5;{fg_color}m"
    bg = f"\x1b[48;5;{bg_color}m"
    return fg, bg

def show_colors():
    
    for i in range(256):
        cstr = f"\x1b[38;5;{i}m"
        print(cstr, i, SCALE, "this is a color!", SCALE[::-1], Colors.RESET)
    print()
    for i in range(256):
        cstr = f"\x1b[48;5;{i}m"
        print(cstr, i, SCALE, "this is a color!", SCALE[::-1], Colors.RESET)
    
def show_colors_24b(min, max, step):
    
    for i in range(min, max, step):
        r = i % 256
        g = (i//256) % 256
        b = (i//(256*256)) % 256
        
        cstr = f"\x1b[38;2;{r};{g};{b}m"
        print(cstr, i, SCALE, "this is a color!", SCALE[::-1], Colors.RESET)
    print()
    for i in range(min, max, step):
        r = i % 256
        g = (i//256) % 256
        b = (i//(256*256)) % 256
        
        cstr = f"\x1b[48;2;{r};{g};{b}m"
        print(cstr, f"[{r},{g},{b}]", SCALE, "this is a color!", SCALE[::-1], Colors.RESET)
    
    print(repr(cstr))

def get_ansi_color(col, bg = False):
    
    if bg:
        opt = 48
    else:
        opt = 38
    
    if isinstance(col, list):
        r,g,b = col
        code = f"\x1b[{opt};2;{r};{g};{b}m"
    else:
        code = f"\x1b[{opt};5;{col}m"
    
    return code
    
def scalar_to_text_8b(scalars, minval = None, maxval = None, fg_color = 53, bg_color = 234, flip = False):
    return scalar_to_text_nb(scalars, minval = minval, maxval = maxval, fg_color = fg_color, bg_color = bg_color, bit_depth = 8, flip = flip)

def scalar_to_text_16b(scalars, minval = None, maxval = None, fg_color = 53, bg_color = 234, flip = False):
    return scalar_to_text_nb(scalars, minval = minval, maxval = maxval, fg_color = fg_color, bg_color = bg_color, bit_depth = 16, flip = flip)

def scalar_to_text_nb(scalars, minval = None, maxval = None, fg_color = 53, bg_color = 234, bit_depth = 24, flip = False, effect = None, add_range = False, **kwargs):
    """
    ok hear me out: plot two quantities together by putting the larger "behind" the smaller using bg. e.g.:
    wait this doesn't work, two idp quantities cant share the same cell
    {bg0fg0}██{bg=fg2 fg0}█
    
    okay how about: denser labels using longer lines, like
        ▁▃▂▂█▅
       1╯││││╰6
        4╯││╰12   
         2╯╰3
    
    """
    
    if flip:
        bg, fg = get_fgbg(bg_color, fg_color)
    else:
        fg, bg = get_fgbg(fg_color, bg_color)
    
    add_border = False
    eff = ""
    if effect:
        if effect == "border":
            add_border = True
        else:
            eff += str(effect)
    
    
    base_bit_depth = len(SCALE) - 1
    if not bit_depth % base_bit_depth == 0:
        return ["no"]
    
    nrows = bit_depth // base_bit_depth
    ncols = len(scalars)
    nvals = base_bit_depth * nrows
    
    rows = [[fg+bg+eff] for r in range(nrows)]
    
    bit_ranges = [base_bit_depth*i for i in range(nrows)]
    
    if minval is None:
        minval = min(scalars)
    if maxval is None:
        maxval = max(scalars)
    rng = (maxval - minval)/1
    c = (minval+ maxval)/2
    if rng == 0:
        # return [fg+bg+eff + SCALE[-1] + RESET]
        rng = 1

    
    for s in scalars:
        sv = int(nvals*((s - c)/rng)) + bit_depth // 2
            
        for row, bit_range in zip(rows, bit_ranges):
            if sv < bit_range:
                sym = SCALE[0]
            elif sv >= bit_range + base_bit_depth:
                sym = SCALE[-1]
            else:
                ssv = sv % base_bit_depth
                sym = SCALE[ssv]
            row.append(sym)
    
    brdc = "\x1b[38;5;6m"
    outstrs= []
    for row in rows[::-1]:
        if add_border:
            row.insert(0, brdc + OTHER.get("right_eighth",""))
            row.append(brdc + SCALE_H[1])
        row.append(RESET)
        outstrs.append("".join(row))
    
    if add_border:
        outstrs.insert(0, " " + SCALE[1]*ncols + " ")
        outstrs.append(f"{brdc} " + OTHER.get("upper_eighth","")*ncols + f" {RESET}")
    
    if add_range:
        # hilo = "⎴⎵"
        # hilo = "⏋⏌"
        ran_fstr = kwargs.get("range_fstr", "<5.2f")
        hilo = "⌝⌟"
        hi, lo = list(hilo)
        hi, lo = (lo, hi) if flip else (hi, lo)
        minstr = format(minval, ran_fstr)
        maxstr = format(maxval, ran_fstr)
        outstrs[0] += hi + maxstr
        if bit_depth > 8:
            outstrs[-1] += lo + minstr
            for i in range(1, len(outstrs) - 1):
                outstrs[i] += " "*max(len(hi+maxstr),len(lo+minstr))
        
    if flip:
        outstrs = flip_scalar_text(outstrs)
    
    return outstrs

def scalar_to_text_mid(scalars, center = None, rng = None, fg_color = 53, bg_color = 234,  effect = None):
    
    bit_depth = 16
    
    ifg, ibg = get_fgbg(fg_color, bg_color)
    bg, fg = get_fgbg(bg_color, fg_color)
    
    eff = ""
    if effect:
        eff += str(effect)
    
    base_bit_depth = len(SCALE) - 1
    if not bit_depth % base_bit_depth == 0:
        return ["no"]
    
    nrows = bit_depth // base_bit_depth
    ncols = len(scalars)
    nvals = base_bit_depth * nrows
    
    rows = [[fg+bg+eff],[ifg+ibg+eff]]
    
    bit_ranges = [base_bit_depth*i - bit_depth/2 for i in range(nrows)]
    
    if not center:
        c = 0
    else:
        c = center
    
    if not rng:
        minval = min(scalars)
        maxval  = max(scalars)
        rng = 2*max(abs(minval), abs(maxval))
    minval, maxval = c-rng/2, c+rng/2
    
    neg = False
    for s in scalars:
        sv = int(nvals*((s - c)/rng))
        if sv < 0 and not neg:
            neg = True
        elif sv >= 0 and neg:
            neg = False
        
        for row, bit_range in zip(rows, bit_ranges):
            if sv < bit_range:
                sym = SCALE[0]
            elif sv >= bit_range + base_bit_depth:
                sym = SCALE[-1]
            else:
                ssv = sv % base_bit_depth
                sym = SCALE[ssv]
            row.append(sym)
    
    outstrs= []
    for row in rows[::-1]:
        row.append(RESET)
        outstrs.append("".join(row))
        
    return outstrs

def add_ruler(sctxt, xmin, xmax, genomic = False, **kwargs):
    num_cols = sum(1 for s in sctxt[0] if s in SCALE)
    ran = (xmax - xmin)
    num_labels = kwargs.get("num_labels", 5)
    ticks = kwargs.get("ticks", 5)
    minor_ticks = kwargs.get("minor_ticks", num_cols)
    lbl_delta = ran / (num_labels - 1)
    if kwargs.get("fstr"):
        fmtr = kwargs.get("fstr")
    elif genomic:
        nexp = np.log10(xmax)
        if nexp > 6:
            div = 1e6
            unit = "M"
        elif nexp > 3:
            div = 1e3
            unit = "k"
        else:
            div = 1
            unit = "b" 
        fmtr = lambda x: format(x/div,".1f") + unit 
    elif any(abs(x)>1e5 for x in [xmin, xmax]):
        fmtr = "0.2e"
    elif any(abs(x) < 1e-5 for x in [xmin, xmax, lbl_delta]):
        fmtr = "0.2e"
    elif all(int(x)==x for x in [xmin, xmax, lbl_delta]):
        fmtr = ".0g"
    else:
        fmtr = "0.1f"
    
    ruler, dists = make_ruler(xmin, xmax, num_cols, num_labels = num_labels, ticks = ticks, minor_ticks = minor_ticks, formatter = fmtr)
    sctxt.append(ruler)

    return sctxt, dists

def make_ruler(xmin, xmax, num_cols, num_labels = 5, ticks = 5, minor_ticks = 5, formatter = "0.2g"):
    
    xran = xmax - xmin
    
    num_labels = max(2, num_labels)
    
    if isinstance(formatter, str):
        frmstr = formatter
        formatter = lambda s: format(s, frmstr)
    
    label_dist_c = round(num_cols / (num_labels - 1))
    if ticks < 1:
        tick_dist_c = num_cols + 1
    else:
        tick_dist_c = round(num_cols / ticks / (num_labels - 1))
    
    if minor_ticks < 1:    
        minor_tick_dist_c = num_cols + 1
    else:
        minor_tick_dist_c = max(1, round(num_cols / minor_ticks / max(1,ticks) / (num_labels - 1)))
    
    label_dist = xran * label_dist_c / num_cols
    tick_dist = xran * tick_dist_c / num_cols
    minor_tick_dist = xran * minor_tick_dist_c / num_cols
    
    bfr = ""
    if tick_dist_c < 2 or minor_tick_dist_c < 2:
        bfr = ""
    
    lbl = "╰"
    rlbl = "╯"
    tck = "╵"
    mtck = "'"
    
    final_lbl = bfr + formatter(xmax) + rlbl
    final_lbl_pos = num_cols - len(final_lbl)
    
    ruler = []
    nc = 0
    while nc < num_cols + 1:
        
        if nc == final_lbl_pos:
            ruler.append(final_lbl)
            break
        elif nc%label_dist_c == 0:
            labelpos = nc*xran/num_cols + xmin
            labelstr = lbl + formatter(labelpos) + bfr
            ruler.append(labelstr)
            nc += len(labelstr)
            continue
        elif nc%tick_dist_c == 0:
            ruler.append(tck)
            nc+= 1
            continue
        elif nc%minor_tick_dist_c == 0:
            ruler.append(mtck)
            nc+= 1
            continue
        else:
            ruler.append(" ")
            nc += 1
    
    return "".join(ruler), (label_dist, tick_dist, minor_tick_dist)

def scalar_plot_distribution(dist_dict, key_order = [], bit_depth = 8, labels = False, labelfmt = "", edges = False, **kwargs):
    
    if not key_order:
        key_order = sorted(dist_dict.keys())
    scalars = [dist_dict[ks] for ks in key_order]
    
    _ = kwargs.pop("minval",None)
    sctxt = scalar_to_text_nb(scalars, bit_depth = bit_depth, minval = 0, **kwargs)
    
    if edges:
        clr = "\x1b[38;5;240m"
        lft, rgt = list(leftright)
        crgt = "{}{}{}".format(clr, rgt, RESET)
        clft = "{}{}{}".format(clr, lft, RESET)
        
        for i in range(len(sctxt)):
            sctxt[i] = clft + sctxt[i] + crgt
    
    if labels:
        if not labelfmt:
            labelfmtr = lambda a:str(a)[0].upper()
        else:
            labelfmtr = lambda a:format(a,labelfmt)
        lbls = [labelfmtr(k) for k in key_order]
        sctxt.append(lbls)
    
    return sctxt

def scrub_ansi(line):
    
    import re
    ansi_re = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    newline = ansi_re.sub("", line)
    return newline
    
def plot_adjacent(seqa, seqb):
    bit_depth = 16
    qseqa = quantize(seqa, bit_depth, mid=True)
    qseqb = quantize(seqb, bit_depth, mid=True)
    
    min_offset = 0
    for a, b in zip(qseqa, qseqb):
        off = a-b
        if off < min_offset:
            min_offset = off
    min_offset -= 1
    eff_bit_depth = bit_depth - min_offset
    
    
    return qseqa, [b+min_offset for b in qseqb], min_offset, eff_bit_depth

def quantize(data, bit_depth, maxval=None, minval=None, mid = False):
    if not maxval:
        maxval = max(data)
    if not minval:
        minval = min(data)
    rng = maxval-minval
    c = (minval+maxval)/2
    off = 0 if mid else 0.5
    return [int(bit_depth * (((d-c)/rng) + off)) for d in data]

def flip_scalar_text(sctext):
    
    nsyms = len(SCALE)
    scale_inv = {SCALE[i]:SCALE[nsyms-i-1] for i in range(nsyms)}
    
    out = []
    for row in sctext[::-1]:
        newrow = []
        for sym in row:
            newrow.append(scale_inv.get(sym, sym))
        out.append("".join(newrow))
    return out

def clean_scalar_text(sctext):
    if isinstance(sctext, str):
        sctext = [sctext]
    elif not isinstance(sctext, list):
        sctext = list(sctext)
    
    outrows = []
    for row in sctext:
        newrow = [t for t in row if t in SCALE]
        outrows.append("".join(newrow))
    return outrows

def show_history(genome, history, t, show_product = False):
    
    hdrs = ["Gene"]
    rows = [[gene_name] for gene_name in genome.gene_order]
    
    for tt in range(t+1):
        hdrs.append(f"Expr(t={tt})")
        if show_product:
            hdrs.extend([f"Prod(t={tt}, chr={n})" for n in range(genome.ploidy)])
        
        expr = history["expression"][tt]
        prod = history["product"][tt]
        for ng in range(genome.num_genes):
            rows[ng].append(expr[ng])
            if show_product:
                for nch in range(genome.ploidy):
                    rows[ng].append(prod[nch][ng])
    
    print(tabulate(rows, headers = hdrs, floatfmt = "0.3f"))
    print()
    
def plot_expression(org, genes = False, phenes = True):
    plot_expressions(org, genes=genes, phenes=phenes)
    
def plot_interesting_genes(*orgs, topk = 5, headers = []):
    
    ngs = orgs[0].genome.num_genes - orgs[0].genome.num_phenes
    scores = np.zeros((ngs,))
    
    for org in orgs:
        ts = org.get_time_series()[:,:ngs].T
        
        covs:np.ndarray = np.cov(ts)
        exc_cov = covs / covs.diagonal()
        
        scores += covs.diagonal()/np.sum(np.abs(exc_cov), axis = 0)
    
    for ig in range(ngs):
        print(orgs[0].genome.gene_order[ig],format(scores[ig], '0.5f'))
    
    best = sorted(enumerate(scores), key = lambda k:-k[1])[:topk]
    
    gene_names = [orgs[0].genome.gene_order[i] for i,_ in best]
    
    plot_expressions(*orgs, genes = True, phenes = False, headers = headers, gene_names = gene_names)
    

def plot_expressions(*orgs, genes = False, phenes = True, gene_names = []):
    
    rowfmt = "{:<16}{}"
    delimit = "     "
    cols = [53, 136]
    
    names = []
    all_rows = []
    for ng, gi in enumerate(orgs[0].genome):
        
        if not genes and gi.is_genotype:
            continue
        elif gene_names and not gi.name in gene_names:
            continue
        elif not phenes and gi.is_phenotype:
            continue
        
        rows = [[],[],[]]
        for org in orgs:
            
            gene_data = []
            for tt in range(org.t):
                expr = org.history["expression"][tt][ng]
                gene_data.append(expr)
            
            col = cols[1] if gi.is_phenotype else cols[0]
            maxval = None if gi.is_phenotype else 1
            
            sctxt = scalar_to_text_nb(gene_data, minval = 0, maxval = maxval, add_range = True, fg_color = col)
            for i in range(len(sctxt)):
                rows[i].append(sctxt[i])
        names.append(gi._base_name)
        all_rows.append(rows)
        
    headers = [f"Genome{org.genome.tag} ({i})" for i, org in enumerate(orgs)]
    fig_len = orgs[0].t + 5 + len(delimit)
    print(rowfmt.format("", "".join([format(hdr, f"<{fig_len}") for hdr in headers])))
    
    for ng in range(len(all_rows)):
        rows = all_rows[ng]
        lbls = [names[ng], "", ""]
        for lbl,row in zip(lbls,rows):
            print(rowfmt.format(lbl, "    ".join(row)))
        print()
        

def plot_product(genome, history, t):
    
    rowfmt = "{:<8}{}"
    
    for ng in range(genome.num_genes):
        gene_name = genome.gene_order[ng]
        for nchr in range(genome.ploidy):
            gene_data = []
            for tt in range(t+1):
                prod = history["product"][tt][nchr][ng]
                gene_data.append(prod)

            sctxt = scalar_to_text_nb(gene_data, minval = 0, add_range = True)
            lbls = [genome.genotype[nchr][gene_name].id, "", ""]
            for r, lbl in zip(sctxt, lbls):
                print(rowfmt.format(lbl, r))
            print()

def plot_states(genome, history):
    
    rowfmt = "{:<8}{}"
    
    for t in range(t):
        
        state_data = []
        for ng in range(genome.num_genes):
            gexpr = history["expression"][t][ng]
            state_data.append(gexpr)
            
        sctxt = scalar_to_text_nb(state_data, minval = 0, add_range = True)
        lbls = [f"t={t}", "", ""]
            
        for r, lbl in zip(sctxt, lbls):
            print(rowfmt.format(lbl, r))
        print()

def show_genome(genome):
    
    hdrs = ["Gene", "Expression", "Regulation", "# Downstream", "# Upstream", "Product", "Scale", "Threshold", "Decay"]
    rows = []
    
    num_upstream = genome.count_upstream()
    num_downstream = genome.count_downstream()
    
    rows = []
    for gi in range(genome.num_genes):
        gene_name = genome.gene_order[gi]
        g = genome.genes[gene_name]

        row = [g.name, format(genome.expression.get(g.name, 0.0), "0.3f"), format(g.regulation, "0.3f")]
        row.extend([num_downstream[gi], num_upstream[gi]])
        row.extend(["--" for i in range(4)])
        rows.append(row)
        
        for nch in range(genome.ploidy):
            a = genome.genotype[nch].get(g.name)
            row = [a.id, "--", "--", "--", "--", format(a.product,"0.3f"), format(a.scale,"0.3f"), format(a.threshold,"0.3f"), format(a.decay,"0.3f")]
            rows.append(row)
    
    print(tabulate(rows, headers = hdrs, floatfmt = "0.3f"))
    print()

def show_interactions(genome):

    tab = []
    headers = [gj.name for gj in genome.iter_genes()]

    for i, gi in enumerate(genome):
        row = [gi.name]
        gi_inters = genome.interactions.get(gi.name, {})
        for j, gj in enumerate(genome.iter_genes()):
            inter = gi_inters.get(gj.name)
            if not inter:
                inter_wgt = 0.0
            else:
                inter_wgt = inter.weight if hasattr(inter, "weight") else f"{inter}(!)"
            row.append(inter_wgt)
        tab.append(row)

    print("Interactions (cols are effectors)")
    print(tabulate(tab, headers = headers, floatfmt = "0.3f"))

def show_interaction_heatmap(inter_arr, col_labels = [], **kwargs):
    heatmap(inter_arr, col_labels = col_labels, **kwargs)


def heatmap(data, row_labels=None, col_labels=None, minval=None, maxval=None,
            center=0, color_scheme = "terra",
            show_values=False, value_fmt="0.2f", colorbar=True):

    if isinstance(data, np.ndarray):
        data = data.tolist()

    min_color, max_color = get_color_scheme_24b(color_scheme)
    mid_color, _ = get_color_scheme_24b("vscode")
    
    nrows = len(data)
    ncols = len(data[0]) if nrows > 0 else 0

    if nrows == 0 or ncols == 0:
        print("Empty data")
        return

    flat_data = [val for row in data for val in row]

    if minval is None:
        minval = min(flat_data)
    if maxval is None:
        maxval = max(flat_data)
        
    rng = maxval - minval
    if rng == 0:
        rng = 1
    
    colors = Colors()

    if center is not None:
        sym_range = max(abs(maxval - center), abs(minval - center))
        minval = center - sym_range
        maxval = center + sym_range

        num_colors = 20
        cs1 = colors.get_color_scale_24b(mid_color, min_color, num_colors//2)
        cs2 = colors.get_color_scale_24b(mid_color, max_color, num_colors//2)
        color_scale = cs1[::-1] + cs2
    else:
        if min_color is None:
            min_color = 16
        if max_color is None:
            max_color = 226

        num_colors = 20
        color_scale = colors.get_color_scale(min_color, max_color, num_colors)

    def value_to_color(val):
        normalized = (val - minval) / rng
        color_idx = int(normalized * (len(color_scale) - 1))
        color_idx = max(0, min(len(color_scale) - 1, color_idx))
        return color_scale[color_idx]

    if row_labels is None:
        row_labels = []
    if col_labels is None:
        col_labels = []
        
    max_row_label_len = max(len(str(lbl)) for lbl in row_labels) if row_labels else 0

    if col_labels:
        header_line = " " * (max_row_label_len + 2)
        for col_lbl in col_labels:
            short_lbl = str(col_lbl)[:2].center(2)
            header_line += short_lbl
        print(header_line)

    block = SCALE[-1]

    for i, row in enumerate(data):
        row_lbl = str(row_labels[i]).ljust(max_row_label_len) if row_labels else ""
        line = row_lbl + "  "

        for val in row:
            color_code = value_to_color(val)
            fg = get_ansi_color(color_code)
        
            if show_values:
                val_str = format(val, value_fmt)
                fg = get_ansi_color(232) if color_code > 128 else get_ansi_color(20)
                line += fg + fg + val_str[:2].center(2) + RESET
            else:
                line += fg + block * 2 + RESET

        print(line)

    if colorbar:
        print()
        colorbar_len = 40
        colorbar_line = " " * (max_row_label_len + 2)

        for i in range(colorbar_len):
            normalized = i / (colorbar_len - 1)
            val = minval + normalized * rng
            color_code = value_to_color(val)
            fg = get_ansi_color(color_code)
            colorbar_line += fg + SCALE[-1] + RESET

        print(colorbar_line)

        # Colorbar labels
        label_line = " " * (max_row_label_len + 2)
        label_line += format(minval, value_fmt).ljust(colorbar_len // 2)
        label_line += format(maxval, value_fmt).rjust(colorbar_len // 2)
        print(label_line)

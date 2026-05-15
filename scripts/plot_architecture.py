"""
SE-RAG System Architecture Diagram
Offline mining + Online reasoning, two-phase structure
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.5)
ax.axis('off')

# ============ Colors ============
C_OFFLINE = '#E8F5E9'     # light green bg
C_ONLINE  = '#E3F2FD'     # light blue bg
C_BOX     = '#FFFFFF'
C_BORDER  = '#333333'
C_GREEN   = '#2E7D32'
C_BLUE    = '#1565C0'
C_ORANGE  = '#E65100'
C_GRAY    = '#757575'
C_ARROW   = '#555555'

# ============ Background Phase Boxes ============
# Offline Phase (left)
offline_rect = FancyBboxPatch((0.1, 4.0), 4.8, 3.2,
                              boxstyle="round,pad=0.15",
                              facecolor=C_OFFLINE, edgecolor=C_GREEN, linewidth=2, alpha=0.5)
ax.add_patch(offline_rect)
ax.text(0.3, 6.9, 'Offline Phase', fontsize=11, fontweight='bold',
        color=C_GREEN, va='center', ha='left')

# Online Phase (right)
online_rect = FancyBboxPatch((5.1, 0.5), 4.8, 4.0,
                             boxstyle="round,pad=0.15",
                             facecolor=C_ONLINE, edgecolor=C_BLUE, linewidth=2, alpha=0.5)
ax.add_patch(online_rect)
ax.text(5.3, 4.25, 'Online Phase', fontsize=11, fontweight='bold',
        color=C_BLUE, va='center', ha='left')

# ============ Helper function ============
def draw_box(ax, x, y, w, h, text, color='white', edge='#333', fontsize=9,
             sub_text=None, sub_color='#666', sub_size=7, align='center'):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, fontsize=fontsize, fontweight='bold',
            color='#222', va='center', ha=align, transform=ax.transData)
    if sub_text:
        ax.text(x + w/2, y + h/2 - 0.18, sub_text, fontsize=sub_size,
                color=sub_color, va='top', ha=align, transform=ax.transData)

def draw_arrow(ax, x1, y1, x2, y2, color='#555', style='-', lw=1.5):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, linestyle=style),
                annotation_clip=False)

def draw_arrow_label(ax, x, y, text, fontsize=7, color='#666'):
    ax.text(x, y, text, fontsize=fontsize, color=color,
            ha='center', va='bottom', style='italic',
            transform=ax.transData)

# ============ OFFLINE PHASE BOXES ============
# Historical Cases (top)
draw_box(ax, 0.8, 5.7, 3.6, 0.55, 'Historical Cases\n(30k subset)',
         color='#F1F8E9', edge=C_GREEN, fontsize=8,
         sub_color='#888', sub_size=6)

# Rule Mining
draw_box(ax, 0.8, 4.5, 1.6, 0.5, 'Rule Mining', color='#C8E6C9', edge=C_GREEN,
         fontsize=9, sub_text='(rule_miner.py)', sub_color=C_GREEN, sub_size=6)

# Causal Extraction
draw_box(ax, 2.8, 4.5, 1.6, 0.5, 'Causal Extraction', color='#C8E6C9', edge=C_GREEN,
         fontsize=9, sub_text='(causal_extractor.py)', sub_color=C_GREEN, sub_size=6)

# Rule Base (bottom-left offline)
draw_box(ax, 0.8, 3.0, 1.6, 0.5, 'Rule Base\n(44k rules)', color='#A5D6A7', edge=C_GREEN,
         fontsize=9, sub_color='#666', sub_size=7)

# Causal Graph (bottom-right offline)
draw_box(ax, 2.8, 3.0, 1.6, 0.5, 'Causal Graph', color='#A5D6A7', edge=C_GREEN,
         fontsize=9, sub_text='(causal_extractor.py)', sub_color=C_GREEN, sub_size=6)

# Arrows in offline phase
draw_arrow(ax, 2.6, 5.7, 2.6, 5.0)  # Cases → split
draw_arrow(ax, 1.6, 5.0, 1.6, 5.0)  # invisible midpoint
ax.plot(2.6, 5.02, 'o', color=C_GREEN, markersize=3)  # split dot
# split to left and right
draw_arrow(ax, 2.6, 5.0, 1.6, 5.0)
draw_arrow(ax, 1.6, 5.0, 1.6, 4.5)  # to rule mining
draw_arrow(ax, 2.6, 5.0, 3.6, 5.0)
draw_arrow(ax, 3.6, 5.0, 3.6, 4.5)  # to causal extraction
# rule mining to rule base
draw_arrow(ax, 1.6, 4.5, 1.6, 3.5)
# causal extraction to causal graph
draw_arrow(ax, 3.6, 4.5, 3.6, 3.5)

# ============ ONLINE PHASE BOXES ============
# Input Event (top-right)
draw_box(ax, 6.2, 3.8, 2.6, 0.45, 'New Disruption Event',
         color='#FFF3E0', edge=C_ORANGE, fontsize=9)

# Feature Extraction
draw_box(ax, 6.2, 3.0, 2.6, 0.45, 'Feature Extraction\n(24-dim vector)',
         color='#FFECB3', edge=C_ORANGE, fontsize=8)

# Rule Matching (decision diamond)
draw_box(ax, 6.2, 2.1, 2.6, 0.45, 'Rule Matching', color='#BBDEFB', edge=C_BLUE,
         fontsize=9, sub_text='(IF-THEN matching)', sub_color=C_BLUE, sub_size=6)

# Fallback: Case Retrieval
draw_box(ax, 8.3, 1.2, 1.5, 0.45, 'Case\nRetrieval', color='#E1BEE7', edge='#7B1FA2',
         fontsize=8, sub_text='(BM25+fallback)', sub_color='#7B1FA2', sub_size=5)

# LLM + Rules + Causal reasoning
draw_box(ax, 5.7, 1.2, 2.3, 0.55, 'LLM + Rules + Causal Reasoning',
         color='#90CAF9', edge=C_BLUE, fontsize=8,
         sub_text='(causal-enhanced inference)', sub_color=C_BLUE, sub_size=6)

# Final output box
output_box = FancyBboxPatch((5.7, 0.2), 2.8, 0.45,
                            boxstyle="round,pad=0.08",
                            facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2.5)
ax.add_patch(output_box)
ax.text(7.1, 0.425, 'Dispatch Decision (Action)', fontsize=9, fontweight='bold',
        color='#222', va='center', ha='center')

# Arrows in online phase
draw_arrow(ax, 7.5, 3.8, 7.5, 3.45)  # event → feature extraction
draw_arrow(ax, 7.5, 3.0, 7.5, 2.55)  # feature extraction → rule matching

# Rule matching → LLM (matched: top path)
draw_arrow(ax, 7.5, 2.1, 6.85, 1.75)  # to LLM
draw_arrow_label(ax, 7.2, 1.85, 'Matched', fontsize=6, color=C_BLUE)

# Rule matching → Case retrieval (unmatched: bottom path)
draw_arrow(ax, 8.8, 2.1, 9.05, 1.65, color='#7B1FA2')  # to case retrieval
draw_arrow_label(ax, 9.0, 1.85, 'Unmatched', fontsize=6, color='#7B1FA2')

# Case retrieval → LLM
draw_arrow(ax, 9.05, 1.2, 8.0, 1.2, color='#7B1FA2')
draw_arrow(ax, 8.0, 1.2, 8.0, 1.75)  # up to LLM

# LLM → Output
draw_arrow(ax, 7.1, 1.2, 7.1, 0.65, lw=2)

# ============ Arrows: Offline → Online (rule base + causal graph to rule matching) ============
draw_arrow(ax, 1.6, 3.0, 1.6, 0.6, style='--', color=C_GREEN)
draw_arrow(ax, 1.6, 0.6, 5.7, 0.6, style='--', color=C_GREEN)
draw_arrow(ax, 5.7, 0.6, 5.7, 1.2, style='--', color=C_GREEN)

draw_arrow(ax, 3.6, 3.0, 3.6, 0.3, style='--', color=C_GREEN)
draw_arrow(ax, 3.6, 0.3, 5.5, 0.3, style='--', color=C_GREEN)
draw_arrow(ax, 5.5, 0.3, 7.5, 0.3, style='--', color=C_GREEN)
draw_arrow(ax, 7.5, 0.3, 7.5, 0.65, style='--', color=C_GREEN)

# Labels for the offline→online arrows
ax.text(4.3, 0.35, 'Rules/Causal Graph loaded at startup', fontsize=6,
        color=C_GREEN, ha='center', va='bottom', style='italic',
        transform=ax.transData)

# Legend / Summary Metrics at bottom
metrics_text = (
    "SE-RAG Key Metrics (1000 queries)\n"
    "Multi-GT: 88.6%  |  Single: 62.5%  |  Latency: 5.74s  |  Error: 0%"
)
ax.text(5.0, -0.15, metrics_text, fontsize=7.5, color='#444',
        ha='center', va='top', family='monospace',
        transform=ax.transData,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                  edgecolor='#CCC', alpha=0.8))

plt.tight_layout()
plt.savefig('D:/code/logistic-ai/paper/fig_se_rag_architecture.png', dpi=300,
            bbox_inches='tight', pad_inches=0.1)
plt.savefig('D:/code/logistic-ai/paper/fig_se_rag_architecture.pdf',
            bbox_inches='tight', pad_inches=0.1)
print("Done. Architecture diagram saved: paper/fig_se_rag_architecture.png + .pdf")

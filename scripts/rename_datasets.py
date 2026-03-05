#!/usr/bin/env python3
"""
Rename simulated datasets with biologically meaningful names
based on each dataset's dominant topic composition.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "sim_store_v1.sqlite"

TOPIC_TISSUE_MAP = {
    "topic_1":  ("Adipose", "Fatty Acid Metabolism"),
    "topic_2":  ("Thymus", "T Cell Development"),
    "topic_3":  ("Liver", "Redox Homeostasis"),
    "topic_4":  ("Lung", "Epithelial-Mesenchymal Transition"),
    "topic_5":  ("Smooth Muscle", "Cytoskeletal Organization"),
    "topic_6":  ("Spleen", "Kinase Signaling"),
    "topic_7":  ("Prostate", "Apoptosis and Growth Factor Signaling"),
    "topic_8":  ("Liver", "Peroxisomal Lipid Processing"),
    "topic_9":  ("Adrenal Gland", "Steroid and Lipid Biosynthesis"),
    "topic_10": ("Liver", "Coagulation Cascade"),
    "topic_11": ("Skeletal Muscle", "Proteasome and Stress Response"),
    "topic_12": ("Bone Marrow", "DNA Replication"),
    "topic_13": ("Bone Marrow", "Cell Cycle Regulation"),
    "topic_14": ("Skin", "Extracellular Matrix Remodeling"),
    "topic_15": ("Intestine", "Epithelial Homeostasis"),
    "topic_16": ("Brain", "Synaptic Transmission"),
    "topic_17": ("Liver", "Fatty Acid Oxidation"),
    "topic_18": ("Embryonic Stem Cell", "Chromatin Remodeling"),
    "topic_19": ("Cartilage", "Glycosaminoglycan Biosynthesis"),
    "topic_20": ("Developing Limb", "Hedgehog Signaling"),
    "topic_21": ("Fetal Liver", "Erythroid Differentiation"),
    "topic_22": ("Fibroblast", "Wound Response"),
    "topic_23": ("Synovial Tissue", "Inflammatory Response"),
    "topic_24": ("Lymph Node", "Cytokine Signaling"),
    "topic_25": ("Peritoneum", "Innate Immune Activation"),
    "topic_26": ("Spleen", "Antigen Presentation"),
    "topic_27": ("PBMC", "Interferon Response"),
    "topic_28": ("Skin", "Keratinocyte Differentiation"),
    "topic_29": ("Kidney", "Autophagy"),
    "topic_30": ("Vascular", "Cytoskeletal Dynamics"),
    "topic_31": ("Liver", "Cholesterol Metabolism"),
    "topic_32": ("HeLa Cell", "RNA Processing"),
    "topic_33": ("Melanocyte", "MAPK Signaling"),
    "topic_34": ("Heart", "Cardiac Muscle Contraction"),
    "topic_35": ("Intestinal Crypt", "Notch-Wnt Signaling"),
    "topic_36": ("Heart", "Oxidative Phosphorylation"),
    "topic_37": ("Hepatocyte", "AMPK Energy Stress"),
    "topic_38": ("Pancreatic Islet", "Endocrine Differentiation"),
    "topic_39": ("Kidney", "mTOR-Autophagy Signaling"),
    "topic_40": ("Liver", "Peroxisome Biogenesis"),
    "topic_41": ("Neuron", "Vesicle Trafficking"),
    "topic_42": ("Retina", "Antioxidant Defense"),
    "topic_43": ("Testis", "Stress and Spermatogenesis"),
    "topic_44": ("Lung", "TGF-beta Signaling"),
    "topic_45": ("Cortex", "Immediate Early Gene Response"),
    "topic_46": ("Embryo", "mRNA Turnover"),
    "topic_47": ("Limb Bud", "Mesenchymal Development"),
    "topic_48": ("Placenta", "Lipid Transport"),
    "topic_49": ("Colon Organoid", "Wnt-Notch Crosstalk"),
    "topic_50": ("Kidney", "Ion Transport"),
}

SPECIES = [
    "Human", "Human", "Human", "Human", "Human", "Human",
    "Mouse", "Mouse", "Mouse", "Mouse", "Mouse",
    "Human", "Human", "Human", "Mouse", "Mouse",
    "Human", "Human", "Mouse", "Human", "Human",
    "Mouse", "Human", "Mouse", "Human", "Human",
    "Human", "Mouse", "Mouse", "Human", "Human",
    "Mouse", "Human", "Human", "Human", "Human",
]


def get_dominant_topics(conn):
    rows = conn.execute("""
        SELECT d.dataset_id,
               sub.topic_id,
               sub.total_w,
               sub.rn
        FROM dataset d
        JOIN (
            SELECT c.dataset_id, ct.topic_id, SUM(ct.weight) as total_w,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.dataset_id
                       ORDER BY SUM(ct.weight) DESC
                   ) as rn
            FROM cell c
            JOIN cell_topic ct ON c.cell_id = ct.cell_id
            GROUP BY c.dataset_id, ct.topic_id
        ) sub ON d.dataset_id = sub.dataset_id AND sub.rn <= 2
        ORDER BY d.dataset_id, sub.rn
    """).fetchall()

    dataset_topics = {}
    for dataset_id, topic_id, weight, rn in rows:
        dataset_topics.setdefault(dataset_id, []).append(topic_id)
    return dataset_topics


def generate_name(dataset_id, topic_ids, idx):
    primary = topic_ids[0]
    tissue, pathway = TOPIC_TISSUE_MAP.get(primary, ("Unknown", "Unknown"))
    species = SPECIES[idx % len(SPECIES)]

    if len(topic_ids) > 1:
        secondary = topic_ids[1]
        _, pathway2 = TOPIC_TISSUE_MAP.get(secondary, ("Unknown", "Unknown"))
        short_p2 = pathway2.split()[0]
        return f"scRNA-seq {species} {tissue} {pathway} with {short_p2} Features"
    return f"scRNA-seq {species} {tissue} {pathway}"


def main():
    conn = sqlite3.connect(DB_PATH)
    dataset_topics = get_dominant_topics(conn)

    updates = []
    for idx, (dataset_id, topics) in enumerate(sorted(dataset_topics.items())):
        name = generate_name(dataset_id, topics, idx)
        updates.append((name, dataset_id))
        print(f"  {dataset_id}: {name}")

    conn.executemany("UPDATE dataset SET dataset_name = ? WHERE dataset_id = ?", updates)
    conn.commit()
    print(f"\nUpdated {len(updates)} datasets.")
    conn.close()


if __name__ == "__main__":
    main()

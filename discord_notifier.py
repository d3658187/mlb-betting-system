"""Discord webhook notifier for MLB betting recommendations."""
from __future__ import annotations

import math
import os
from datetime import date
from typing import Optional

import pandas as pd
import requests


def _format_price(price: float) -> str:
    if pd.isna(price):
        return "N/A"
    try:
        p = int(price)
        return f"{p:+d}" if p > 0 else f"{p}"
    except Exception:
        return str(price)


def _format_prob(prob: float) -> str:
    if pd.isna(prob):
        return "N/A"
    return f"{prob * 100:.1f}%"


def _format_ev(ev: float) -> str:
    if pd.isna(ev):
        return "N/A"
    return f"{ev * 100:.2f}%"


def _build_embeds(reco: pd.DataFrame, target_date: date) -> list[dict]:
    embeds = []
    for _, row in reco.iterrows():
        title = f"{row['team']} vs {row['opponent']}"
        side = "主隊" if row.get("side") == "home" else "客隊"
        embed = {
            "title": title,
            "description": f"推薦：{side} {row['team']}",
            "color": 0x2ECC71,
            "fields": [
                {"name": "預測勝率", "value": _format_prob(row.get("win_prob")), "inline": True},
                {"name": "台彩賠率", "value": _format_price(row.get("price")), "inline": True},
                {"name": "EV", "value": _format_ev(row.get("ev")), "inline": True},
            ],
            "footer": {"text": f"MLB 高EV推薦 {target_date.isoformat()}"},
        }
        embeds.append(embed)
    return embeds


def send_discord_recommendations(reco: pd.DataFrame, target_date: date) -> bool:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return False

    if reco is None or reco.empty:
        return False

    embeds = _build_embeds(reco, target_date)
    if not embeds:
        return False

    # Discord allows up to 10 embeds per message
    for i in range(0, len(embeds), 10):
        chunk = embeds[i : i + 10]
        payload = {
            "content": None,
            "embeds": chunk,
        }
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code >= 300:
            raise RuntimeError(f"Discord webhook failed: {resp.status_code} {resp.text}")

    return True

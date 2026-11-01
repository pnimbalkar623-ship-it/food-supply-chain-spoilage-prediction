"""
fruit_recommendations.py
------------------------
Practical supply-chain storage and handling recommendations for detected fruits.
Based on postharvest biology standards (temperature, relative humidity, ethylene).
Includes vital safety and scientific disclaimers regarding optical imaging limitations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StorageRecommendation:
    fruit_name: str
    optimal_temp: str
    optimal_humidity: str
    ethylene_behavior: str
    storage_tips: list[str]
    max_shelf_life: str


RECOMMENDATION_DATABASE: dict[str, StorageRecommendation] = {
    "apple": StorageRecommendation(
        fruit_name="Apple",
        optimal_temp="0°C to 4°C (32°F to 39°F)",
        optimal_humidity="90% – 95% RH",
        ethylene_behavior="High ethylene producer. Highly sensitive to other ethylene sources.",
        storage_tips=[
            "Store in cold storage with continuous air circulation.",
            "Isolate from ethylene-sensitive produce (e.g., leafy greens, bananas) to prevent premature ripening.",
            "Inspect regularly for surface bruises that accelerate Penicillium (blue mold) infection.",
        ],
        max_shelf_life="1 to 6 months depending on variety and atmosphere control.",
    ),
    "banana": StorageRecommendation(
        fruit_name="Banana",
        optimal_temp="13°C to 15°C (56°F to 59°F)",
        optimal_humidity="90% – 95% RH",
        ethylene_behavior="Climacteric; moderate ethylene producer, highly responsive to ethylene.",
        storage_tips=[
            "CRITICAL: Never store below 12°C (chilling injury causes skin browning and failure to ripen).",
            "Store in well-ventilated ripening rooms or hanging racks to avoid pressure bruising.",
            "Use polyethylene bunch liners or MAP (Modified Atmosphere Packaging) during shipment.",
        ],
        max_shelf_life="1 to 4 weeks (green to ripe stages).",
    ),
    "orange": StorageRecommendation(
        fruit_name="Orange",
        optimal_temp="4°C to 7°C (39°F to 45°F)",
        optimal_humidity="85% – 90% RH",
        ethylene_behavior="Non-climacteric; very low ethylene production.",
        storage_tips=[
            "Maintain moderate humidity; excess moisture promotes green/blue mold (Penicillium digitatum).",
            "Ensure crates are dry and pre-cooled before loading into reefer containers.",
            "Avoid contact with decaying citrus to prevent rapid sporulation spread.",
        ],
        max_shelf_life="3 to 8 weeks depending on variety.",
    ),
    "grape": StorageRecommendation(
        fruit_name="Grape",
        optimal_temp="-0.5°C to 0°C (31°F to 32°F)",
        optimal_humidity="90% – 95% RH",
        ethylene_behavior="Non-climacteric; very low ethylene sensitivity.",
        storage_tips=[
            "Maintain near-freezing temperatures without dropping below berry freezing point (-1.5°C).",
            "Utilize SO2 (sulfur dioxide) generator pads in cartons to prevent Botrytis cinerea (gray mold).",
            "Handle strictly by stems to preserve the protective natural bloom on the grape skin.",
        ],
        max_shelf_life="2 to 8 weeks.",
    ),
    "pineapple": StorageRecommendation(
        fruit_name="Pineapple",
        optimal_temp="7°C to 10°C (45°F to 50°F)",
        optimal_humidity="85% – 90% RH",
        ethylene_behavior="Non-climacteric; low ethylene production.",
        storage_tips=[
            "Chilling sensitive below 7°C (leads to internal browning and water-soaked flesh).",
            "Keep upright in cushioned transport boxes to protect the crown and base.",
            "Ship and market rapidly once harvested at full maturity.",
        ],
        max_shelf_life="2 to 4 weeks.",
    ),
    "watermelon": StorageRecommendation(
        fruit_name="Watermelon",
        optimal_temp="10°C to 15°C (50°F to 59°F)",
        optimal_humidity="85% – 90% RH",
        ethylene_behavior="Sensitive to ethylene; exposure causes softening and rind thinning.",
        storage_tips=[
            "Susceptible to chilling injury if held below 10°C for more than a few days.",
            "Store separate from apples, bananas, and tomatoes.",
            "Avoid stacking too high to prevent pressure cracking in bottom cartons.",
        ],
        max_shelf_life="2 to 3 weeks.",
    ),
}

DEFAULT_RECOMMENDATION = StorageRecommendation(
    fruit_name="General Produce",
    optimal_temp="4°C to 10°C",
    optimal_humidity="85% – 90% RH",
    ethylene_behavior="Varies by fruit variety.",
    storage_tips=[
        "Maintain clean, sanitized refrigeration units.",
        "Ensure prompt removal of any visually spoiled or damaged specimens.",
        "Monitor cold-chain temperature continuously during transit.",
    ],
    max_shelf_life="1 to 3 weeks.",
)


def get_recommendation_for_fruit(fruit_name: str) -> StorageRecommendation:
    """Retrieve verified storage guidelines for a fruit category."""
    clean_key = fruit_name.strip().lower()
    for key, rec in RECOMMENDATION_DATABASE.items():
        if key in clean_key:
            return rec
    return DEFAULT_RECOMMENDATION


def get_scientific_disclaimer() -> str:
    """
    Mandatory scientific limitation statement:
    Ordinary RGB optical images cannot confirm chemical additives (such as formalin)
    or internal microbial pathogens.
    """
    return (
        "⚠️ **Scientific & Inspection Disclaimer:** "
        "Computer vision image analysis is strictly limited to visible external characteristics "
        "(shape, size, surface discoloration, and count). **Standard optical photography CANNOT "
        "confirm the presence of chemical preservatives, artificial ripening agents, formalin, "
        "pesticide residue, or internal microbial infection.** Formal food safety compliance "
        "requires validated biochemical laboratory testing."
    )

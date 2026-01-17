# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI per avviare la simulazione del retailer all'interno di ASSUME World."""

from __future__ import annotations

import logging
from pathlib import Path

from assume.extensions.retailer_sim.data_utils import (
    filter_dataframe_for_simulation,
    load_config,
    load_dataframe,
)
from assume.extensions.retailer_sim.world_adapter.world_simulation import create_world_from_dataframe
from assume.extensions.retailer_sim.output_utils import export_world_results

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Fase 0: recuperiamo configurazione e dataset con gli stessi helper usati
    # nello standalone cosi entrambe le pipeline rimangono allineate.
    config = load_config()
    df = load_dataframe(config)
    df = filter_dataframe_for_simulation(df, config)

    # Fase 1: creiamo il world registrando unita, mercati e la strategia condivisa.
    world = create_world_from_dataframe(df, config)
    # Fase 2: avviamo il loop di simulazione ASSUME (clearing interno del world).
    world.run()

    output_folder = Path(config.get("simulation", {}).get("output_folder", "outputs"))
    logger.info("World simulation finished. Results stored in %s", output_folder.resolve())
    decision_cfg = config.get("decision_making", {})
    strategy_label = str(decision_cfg.get("strategy_type", "simple")).lower()
    export_world_results(config, strategy_label=strategy_label, reference_df=df)


if __name__ == "__main__":
    main()

import asyncio
from typing import Dict, List
from datetime import datetime, timedelta

from fiber.chain import chain_utils, interface, metagraph, weights
from fiber.chain.fetch_nodes import get_nodes_for_netuid
from fiber.logging_utils import get_logger
from validator.db.operations import DatabaseManager
from validator.config import (
    NETUID, 
    SUBTENSOR_NETWORK, 
    SUBTENSOR_ADDRESS,
    WALLET_NAME,
    HOTKEY_NAME,
    VERSION_KEY
)

logger = get_logger(__name__)

async def set_weights(
    db_manager: DatabaseManager,
) -> None:
    """Set weights for miners based on their performance scores from the last 24 hours."""
    try:
        # Get substrate and keypair
        substrate = interface.get_substrate(
            subtensor_network=SUBTENSOR_NETWORK,
            subtensor_address=SUBTENSOR_ADDRESS
        )
        keypair = chain_utils.load_hotkey_keypair(wallet_name=WALLET_NAME, hotkey_name=HOTKEY_NAME)
        
        # Get validator node ID and version key
        validator_node_id = substrate.query("SubtensorModule", "Uids", [NETUID, keypair.ss58_address]).value
        v_key = substrate.query("SubtensorModule", "WeightsVersionKey", [NETUID]).value
        logger.info(f"Subnet Version key: {v_key}")
        version_key = VERSION_KEY
        # Get all active nodes
        nodes = get_nodes_for_netuid(substrate=substrate, netuid=NETUID)
        
        # Get miner scores from the database
        miner_scores = db_manager.get_miner_scores_with_node_id()
        logger.info(f"Fetched scores for {len(miner_scores)} miners")

        # Calculate weights
        node_weights: List[float] = []
        node_ids: List[int] = []
        
        for node in nodes:
            node_id = node.node_id
            score_data = miner_scores.get(node_id, {
                'final_score': 0.0,
                'performance_score': 0.0,
                'speed_score': 0.0,
                'availability_score': 0.0,
                'avg_processing_time': 0.0
            })
            
            node_ids.append(node_id)
            node_weights.append(score_data['final_score'])
        
        # Ensure weights sum to 1.0
        total_weight = sum(node_weights)
        if total_weight > 0:
            node_weights = [w / total_weight for w in node_weights]
        else:
            # If no scores, distribute evenly
            node_weights = [1.0 / len(nodes) for _ in nodes]
        
        # Log detailed weight information
        logger.info(f"Setting weights for {len(nodes)} nodes")
        for node_id, weight, node in zip(node_ids, node_weights, nodes):
            score_data = miner_scores.get(node_id, {
                'final_score': 0.0,
                'performance_score': 0.0,
                'speed_score': 0.0,
                'availability_score': 0.0,
                'avg_processing_time': 0.0
            })
            logger.info(
                f"Node {node_id} ({node.hotkey}): "
                f"perf={score_data['performance_score']:.4f}, "
                f"speed={score_data['speed_score']:.4f}, "
                f"avail={score_data['availability_score']:.4f}, "
                f"final={score_data['final_score']:.4f}, "
                f"avg_time={score_data['avg_processing_time']:.4f}, "
                f"weight={weight:.4f}"
            )
        
        # Set weights on chain
        weights.set_node_weights(
            substrate=substrate,
            keypair=keypair,
            node_ids=node_ids,
            node_weights=node_weights,
            netuid=NETUID,
            validator_node_id=validator_node_id,
            version_key=version_key,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        
        logger.info("Successfully set weights on chain")
        
    except Exception as e:
        logger.error(f"Error setting weights: {str(e)}")
        logger.exception("Full error traceback:")
        raise

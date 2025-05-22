# validator/challenge/challenge_process.py
import asyncio
import os
import time
from datetime import datetime, timezone
from multiprocessing import Process
from validator.challenge.send_challenge import send_challenge
from validator.challenge.challenge_types import GSRChallenge, ChallengeType, ChallengeTask
from validator.utils.api import get_next_challenge
from validator.utils.async_utils import AsyncBarrier
from validator.config import CHALLENGE_INTERVAL
from validator.db.operations import DatabaseManager
from validator.evaluation.evaluation import GSRValidator
from fiber.chain.interface import get_substrate
from validator.main import (
    get_active_nodes_with_stake,
    get_available_nodes,
    construct_server_address,
    process_challenge_results,
)
import httpx
from loguru import logger
from substrateinterface import Keypair


def start_challenge_sender():
    asyncio.run(run_challenge_loop())


async def run_challenge_loop():
    """Loop that sends challenges periodically (in a separate process)."""
    logger.info("[Process] Challenge sender loop starting...")

    hotkey = os.getenv("VALIDATOR_HOTKEY")
    db_manager = DatabaseManager(os.environ["DB_PATH"])
    validator = GSRValidator(openai_api_key=os.environ["OPENAI_API_KEY"], validator_hotkey=hotkey.ss58_address)
    substrate = get_substrate()


    async with httpx.AsyncClient() as client:
        while True:
            try:
                logger.info("Fetching active nodes...")
                nodes = get_active_nodes_with_stake()
                available_nodes = await get_available_nodes(nodes, client, db_manager, hotkey.ss58_address)

                if len(available_nodes) == 0:
                    logger.warning("No available nodes. Sleeping...")
                    await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
                    continue

                challenge_data = await get_next_challenge(hotkey.ss58_address)
                if not challenge_data:
                    logger.warning("No challenge from API. Sleeping...")
                    await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())
                    continue

                logger.info(f"Fetched challenge {challenge_data['task_id']}")
                barrier = AsyncBarrier(parties=len(available_nodes))
                new_challenge_tasks = []

                for node in available_nodes:
                    challenge = GSRChallenge(
                        challenge_id=challenge_data['task_id'],
                        type=ChallengeType.GSR,
                        created_at=datetime.now(timezone.utc),
                        video_url=challenge_data['video_url']
                    )

                    task = asyncio.create_task(
                        send_challenge(
                            challenge=challenge,
                            server_address=construct_server_address(node),
                            hotkey=node.hotkey,
                            keypair=hotkey,
                            node_id=node.node_id,
                            barrier=barrier,
                            db_manager=db_manager,
                            client=client
                        )
                    )

                    challenge_task = ChallengeTask(
                        node_id=node.node_id,
                        task=task,
                        timestamp=datetime.now(timezone.utc),
                        challenge=challenge,
                        miner_hotkey=node.hotkey
                    )
                    new_challenge_tasks.append(challenge_task)


                await process_challenge_results(
                    challenge_tasks=new_challenge_tasks,
                    db_manager=db_manager,
                    validator=validator,
                    keypair=hotkey,
                    substrate=substrate
                )

                await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())

            except Exception as e:
                logger.error(f"[ChallengeProcess] Error: {e}")
                await asyncio.sleep(CHALLENGE_INTERVAL.total_seconds())

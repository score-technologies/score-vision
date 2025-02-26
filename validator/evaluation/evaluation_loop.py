import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Set
from pathlib import Path
import time

import httpx
from fiber.logging_utils import get_logger
from validator.db.operations import DatabaseManager
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse, ValidationResult, GSRChallenge, ChallengeType
from validator.evaluation.calculate_score import calculate_score
from validator.utils.api import update_task_scores
from validator.config import VALIDATION_DELAY, FRAMES_TO_VALIDATE
from validator.evaluation.prompts import COUNT_PROMPT
from loguru import logger
import cv2
from dataclasses import dataclass

# New constant for minimum number of players
MIN_PLAYERS_PER_FRAME = 4
INITIAL_FRAME_SAMPLE_SIZE = 20
RESPONSES_TO_CHECK = 10

# Constants for batching and workers
BATCH_SIZE = 5  # Number of frames to process in one batch
MAX_WORKERS = 5  # Number of concurrent worker tasks
MAX_RETRIES = 3  # Maximum retry attempts for failed batches
WORKER_TIMEOUT = 300  # 5 minutes timeout for worker tasks

logger = get_logger(__name__)

@dataclass
class EvaluationTask:
    """Represents a single evaluation task for a response."""
    response: GSRResponse
    challenge: Dict[str, Any]
    video_path: Path
    frames: List[int]
    frame_cache: Dict
    retry_count: int = 0

class EvaluationQueue:
    """Manages batched evaluation tasks with multiple workers."""
    
    def __init__(self, validator):
        self.validator = validator
        self.queue = asyncio.Queue()
        self.results = {}
        self.active_tasks: Set[asyncio.Task] = set()
        self.frame_cache = {}
        self.processing = False
        
    async def add_task(self, task: EvaluationTask):
        """Add a new evaluation task to the queue."""
        await self.queue.put(task)
        
    async def worker(self, worker_id: int):
        """Worker coroutine that processes evaluation tasks."""
        logger.info(f"Starting worker {worker_id}")
        
        while self.processing:
            try:
                # Get next task
                task = await self.queue.get()
                
                # Process frames in batches
                for i in range(0, len(task.frames), BATCH_SIZE):
                    batch_frames = task.frames[i:i + BATCH_SIZE]
                    try:
                        # Process batch with timeout
                        batch_results = await asyncio.wait_for(
                            self._process_batch(task, batch_frames),
                            timeout=WORKER_TIMEOUT
                        )
                        
                        # Store results
                        response_id = task.response.response_id
                        if response_id not in self.results:
                            self.results[response_id] = []
                        self.results[response_id].extend(batch_results)
                        
                    except asyncio.TimeoutError:
                        logger.error(f"Batch timeout for frames {batch_frames}")
                        if task.retry_count < MAX_RETRIES:
                            # Requeue failed batch
                            retry_task = EvaluationTask(
                                response=task.response,
                                challenge=task.challenge,
                                video_path=task.video_path,
                                frames=batch_frames,
                                frame_cache=task.frame_cache,
                                retry_count=task.retry_count + 1
                            )
                            await self.queue.put(retry_task)
                    except Exception as e:
                        logger.error(f"Batch error: {str(e)}")
                
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)
                
    async def _process_batch(self, task: EvaluationTask, batch_frames: List[int]) -> List[dict]:
        """Process a batch of frames for a task."""
        results = []
        
        # Pre-fetch reference counts for batch if needed
        frames_to_process = []
        for frame_idx in batch_frames:
            if frame_idx not in self.validator.frame_reference_counts:
                if frame_idx in task.frame_cache:
                    frame_data = task.frame_cache[frame_idx]
                    frames_to_process.append({
                        'encoded_image': self.validator.encode_image(frame_data['frame']),
                        'frame_id': frame_idx
                    })
                    
        if frames_to_process:
            reference_results = await self.validator.vlm_processor.get_reference_counts_batch(
                frames_to_process,
                COUNT_PROMPT
            )
            
            for frame_data, result in zip(frames_to_process, reference_results):
                frame_idx = frame_data['frame_id']
                try:
                    if result:
                        counts = json.loads(result)
                        if isinstance(counts, dict):
                            self.validator.frame_reference_counts[frame_idx] = counts
                except Exception as e:
                    logger.error(f"Error processing reference counts: {str(e)}")
        
        # Process each frame in the batch
        for frame_idx in batch_frames:
            try:
                frame = task.frame_cache[frame_idx]['frame']
                frame_data = task.response.frames.get(str(frame_idx), {})
                ref_counts = self.validator.frame_reference_counts.get(frame_idx)
                
                result = await self.validator.validate_frame_detections(
                    frame=frame,
                    detections=frame_data,
                    frame_idx=frame_idx,
                    challenge_id=task.challenge['challenge_id'],
                    node_id=task.response.node_id,
                    reference_counts=ref_counts,
                    response_id=task.response.response_id
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                
        return results
        
    async def start_processing(self):
        """Start worker tasks."""
        self.processing = True
        for i in range(MAX_WORKERS):
            task = asyncio.create_task(self.worker(i))
            self.active_tasks.add(task)
            task.add_done_callback(self.active_tasks.remove)
            
    async def stop_processing(self):
        """Stop all workers and clean up."""
        logger.info("Stopping evaluation queue processing...")
        self.processing = False
        
        if self.active_tasks:
            try:
                # Wait for tasks with a timeout
                done, pending = await asyncio.wait(
                    self.active_tasks,
                    timeout=30  # 30 second timeout
                )
                
                # Cancel any pending tasks
                for task in pending:
                    logger.warning("Cancelling pending worker task that didn't complete in time")
                    task.cancel()
                    
                # Wait briefly for cancelled tasks
                if pending:
                    await asyncio.wait(pending, timeout=5)
                    
            except Exception as e:
                logger.error(f"Error during worker cleanup: {str(e)}")
            finally:
                self.active_tasks.clear()
                logger.info("Evaluation queue cleanup completed")

async def _evaluate_single_response(
    validator: GSRValidator,
    db_manager: DatabaseManager,
    video_path: Path,
    row: sqlite3.Row,
    frame_cache: Dict = None,
    frames_to_validate: List[int] = None
) -> Dict[str, Any]:
    """Evaluate a single response."""
    try:
        # Log row information (excluding response_data)
        row_dict = dict(row)
        if 'response_data' in row_dict:
            row_dict['response_data'] = '<omitted>'
        #logger.info(f"Processing row: {row_dict}")
        
        # Get response data
        response_data = json.loads(row["response_data"] if row["response_data"] else "{}")
        #logger.info(f"Response metadata - challenge_id: {row['challenge_id']}, miner_hotkey: {row['miner_hotkey']}")
        
        # Create GSRResponse object with node_id
        response = GSRResponse(
            challenge_id=row["challenge_id"],
            miner_hotkey=row["miner_hotkey"],
            frames=response_data.get("frames", {}),
            processing_time=row["processing_time"],
            response_id=row["response_id"],
            node_id=row["node_id"]
        )
        #logger.info(f"Created GSRResponse object for response_id: {response.response_id}")

        # Get challenge data
        challenge = GSRChallenge(
            challenge_id=row["challenge_id"],
            type=ChallengeType.GSR,
            created_at=row["created_at"] if "created_at" in row.keys() else None,
            video_url=row["video_url"] if "video_url" in row.keys() else ""
        )
        #logger.info(f"Processing challenge_id: {challenge.challenge_id}")

        # Get timing information from database
        started_at = db_manager.get_challenge_assignment_sent_at(challenge.challenge_id, response.miner_hotkey)
        completed_at = row["completed_at"] if "completed_at" in row.keys() else None
        #logger.info(f"Timing info - started_at: {started_at}, completed_at: {completed_at}")

        # Evaluate response using cached frames if available
        result = await validator.evaluate_response(
            response, 
            challenge, 
            video_path, 
            frame_cache=frame_cache,
            frames_to_validate=frames_to_validate
        )
        logger.info(f"Evaluation complete - score: {result.score}")
        
        # Mark response as evaluated
        db_manager.mark_response_as_evaluated(response.response_id)

        return {
            "challenge_id": challenge.challenge_id,
            "miner_hotkey": response.miner_hotkey,
            "node_id": response.node_id,
            "response_id": response.response_id,
            "score": result.score,
            "processing_time": row["processing_time"],
            "validation_result": result,
            "task_returned_data": response_data,
            "started_at": started_at,
            "completed_at": completed_at,
            "received_at": row["received_at"] if "received_at" in row.keys() else None
        }
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}")
        logger.error(f"Row object type: {type(row)}")
        if hasattr(row, 'keys'):
            logger.error(f"Available row keys: {row.keys()}")
        raise

async def select_frames_with_players(
    validator: GSRValidator,
    video_path: Path,
    db_manager: DatabaseManager,
    challenge_id: str,
    num_frames: int = FRAMES_TO_VALIDATE
) -> List[int]:
    """
    Select frames that contain at least MIN_PLAYERS_PER_FRAME players on average.
    """
    all_frames = validator.select_random_frames(video_path, INITIAL_FRAME_SAMPLE_SIZE)
    frame_cache = {}
    player_counts = {frame: [] for frame in all_frames}

    # Get a sample of responses for this challenge
    responses = db_manager.get_sample_responses(challenge_id, RESPONSES_TO_CHECK)

    for frame in all_frames:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame_image = cap.read()
        cap.release()
        if ret:
            frame_cache[frame] = {'frame': frame_image}

    for response in responses:
        response_data = json.loads(response["response_data"])
        for frame in all_frames:
            frame_data = response_data.get("frames", {}).get(str(frame), {})
            player_count = sum(1 for obj in frame_data.get("objects", []) 
                               if obj["class_id"] in [1, 2])  # goalkeeper and player
            player_counts[frame].append(player_count)

    # Calculate average player count for each frame
    avg_player_counts = {frame: sum(counts) / len(counts) if counts else 0 
                         for frame, counts in player_counts.items()}

    # Select frames with sufficient players
    selected_frames = [frame for frame, avg_count in avg_player_counts.items() 
                       if avg_count >= MIN_PLAYERS_PER_FRAME]

    # If we don't have enough frames, add the ones with the highest average player count
    if len(selected_frames) < num_frames:
        remaining_frames = sorted(
            [f for f in all_frames if f not in selected_frames],
            key=lambda f: avg_player_counts[f],
            reverse=True
        )
        selected_frames.extend(remaining_frames[:num_frames - len(selected_frames)])

    return selected_frames[:num_frames]

async def evaluate_pending_responses(
    validator: GSRValidator,
    db_manager,
    challenge: Dict[str, Any]
) -> None:
    """Evaluate all pending responses for a challenge using the worker pool."""
    eval_queue = None
    try:
        # Initialize evaluation queue
        eval_queue = EvaluationQueue(validator)
        await eval_queue.start_processing()
        
        # Get video path
        video_path = await validator.download_video(challenge['video_url'])
        if not video_path or not video_path.exists():
            logger.error(f"Failed to download video for challenge {challenge['challenge_id']}")
            return

        # Get pending responses
        responses = await db_manager.get_pending_responses(challenge['challenge_id'])
        if not responses:
            logger.info(f"No pending responses for challenge {challenge['challenge_id']}")
            return
        sorted_responses = sorted(responses,key=lambda r: r.processing_time if r.processing_time is not None else float('999'))
        top_responses = sorted_responses[:30]
        evaluation_results = []
        for resp in responses:
            if resp not in top_responses:
                started_at = db_manager.get_challenge_assignment_sent_at(challenge['challenge_id'], resp.miner_hotkey)
                evaluation_results.append({
                    "challenge_id": challenge['challenge_id'],
                    "miner_hotkey": resp.miner_hotkey,
                    "node_id": resp.node_id,
                    "response_id": resp.response_id,
                    "score": 0.4,
                    "processing_time": resp.processing_time,
                    "validation_result": None,
                    "task_returned_data": None,
                    "started_at": started_at,
                    "completed_at": None,  # Will be set by calculate_score
                    "received_at": None  # Will be set by calculate_score
                })
                
        responses=top_responses        
        # Select frames for this challenge
        frames = await select_frames_with_players(
            validator,
            video_path,
            db_manager,
            challenge['challenge_id']
        )
        logger.info(f"Selected {len(frames)} frames for challenge {challenge['challenge_id']}")

        # Pre-load frames into cache
        frame_cache = {}
        for frame_idx in frames:
            if frame_idx not in frame_cache:
                cap = cv2.VideoCapture(str(video_path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_cache[frame_idx] = {'frame': frame}

        # Create and queue tasks for each response
        for response in responses:
            task = EvaluationTask(
                response=response,
                challenge=challenge,
                video_path=video_path,
                frames=frames,
                frame_cache=frame_cache
            )
            await eval_queue.add_task(task)
            logger.info(f"Queued task for response {response.response_id}")

        # Wait for all tasks to complete
        await eval_queue.queue.join()
        
        # Process results
        for response in responses:
            results = eval_queue.results.get(response.response_id, [])
            logger.info(f"Processing results for response {response.response_id}")
            
            if results:
                # Calculate final scores
                frame_scores = {}
                total_score = 0.0
                frame_details = []
                
                for result in results:
                    # Extract frame number from debug frame path
                    frame_path = result.get("debug_frame_path", "")
                    frame_num = None
                    if frame_path:
                        try:
                            frame_num = int(frame_path.split("frame_")[1].split("_")[0])
                        except:
                            logger.error(f"Could not extract frame number from {frame_path}")
                    
                    if frame_num is not None:
                        frame_score = result["scores"]["final_score"]
                        frame_scores[frame_num] = frame_score
                        total_score += frame_score
                        
                        # Only store essential frame details
                        frame_details.append({
                            "frame_number": frame_num,
                            "keypoint_score": result["scores"]["keypoint_score"],
                            "bbox_score": result["scores"]["bbox_score"],
                            "count_match_score": result["scores"]["count_match_score"],
                            "final_score": frame_score,
                            "debug_frame_path": frame_path,
                            "num_objects": len(result.get("objects", [])),
                            "num_keypoints": len(result.get("keypoints", {}).get("points", [])),
                            "reference_counts": result.get("count_validation", {}).get("reference_counts", {}),
                            "detected_counts": result.get("count_validation", {}).get("high_confidence_counts", {})
                        })
                        logger.info(f"Frame {frame_num} score: {frame_score:.3f}")
                
                avg_score = total_score / len(results) if results else 0.0
                logger.info(f"Average score for response {response.response_id}: {avg_score:.3f}")
                
                # Get timing information
                started_at = db_manager.get_challenge_assignment_sent_at(challenge['challenge_id'], response.miner_hotkey)
                
                # Create evaluation result with simplified data
                evaluation_results.append({
                    "challenge_id": challenge['challenge_id'],
                    "miner_hotkey": response.miner_hotkey,
                    "node_id": response.node_id,
                    "response_id": response.response_id,
                    "score": avg_score,
                    "processing_time": response.processing_time,
                    "validation_result": ValidationResult(
                        score=avg_score,
                        frame_scores=frame_scores,
                        feedback=frame_details
                    ),
                    "task_returned_data": response.frames,
                    "started_at": started_at,
                    "completed_at": None,  # Will be set by calculate_score
                    "received_at": None  # Will be set by calculate_score
                })
                
                logger.info(
                    f"Processed response {response.response_id} "
                    f"(score: {avg_score:.3f}, frames: {len(frame_scores)})"
                )
            else:
                logger.error(f"No results for response {response.response_id}")

        # Calculate final scores and update DB/API
        async with httpx.AsyncClient() as client:
            scores = await calculate_score(evaluation_results, client, validator_hotkey=validator.validator_hotkey, db_manager=db_manager)
            
            # Log all scores being processed
            logger.info(f"Processing scores for {len(scores)} responses")
            
            # Update DB and external API for each response
            for response_id, score_data in scores.items():
                node_id = score_data['node_id']
                miner_hotkey = score_data['miner_hotkey']
                
                logger.info(f"Processing response {response_id} for node {node_id}")
                
                # Log detailed scoring information
                logger.info(f"Response {response_id} scoring details:")
                logger.info(f"  - Quality score: {score_data['quality_score']:.3f}")
                logger.info(f"  - Speed score: {score_data['speed_score']:.3f}")
                logger.info(f"  - Availability score: {score_data['availability_score']:.3f}")
                logger.info(f"  - Final score: {score_data['final_score']:.3f}")
                
                # Update response with score and evaluation status
                db_manager.update_response(
                    response_id=response_id,
                    score=score_data['final_score'],
                    evaluated=True,
                    evaluated_at=datetime.utcnow()
                )
                
                db_manager.store_response_score(
                    response_id=response_id,
                    challenge_id=challenge["challenge_id"],
                    validation_result=score_data['validation_result'],
                    validator_hotkey=validator.validator_hotkey,
                    miner_hotkey=miner_hotkey,
                    node_id=int(node_id),
                    availability_score=score_data['availability_score'],
                    speed_score=score_data['speed_score'],
                    total_score=score_data['final_score']
                )
                
                # Update external API for each response
                update_success = await update_task_scores(
                    validator_address=validator.validator_hotkey,
                    task_id=challenge["challenge_id"],
                    challenge_id=challenge["challenge_id"],
                    miner_id=node_id,
                    miner_hotkey=score_data['miner_hotkey'],
                    response_data=json.dumps(score_data['task_returned_data']),
                    evaluation_score=score_data['quality_score'],
                    speed_score=score_data['speed_score'],
                    availability_score=score_data['availability_score'],
                    total_score=score_data['final_score'],
                    processing_time=score_data['processing_time'],
                    started_at=(score_data['started_at']),
                    completed_at=datetime.now(timezone.utc).isoformat()
                )
                
                if update_success:
                    logger.info(f"Successfully updated API with scores for response {response_id}")
                else:
                    logger.warning(f"Failed to update API with scores for response {response_id}")
                
                # Add a small delay between API calls to prevent rate limiting
                await asyncio.sleep(0.5)
            
            logger.info(f"Completed processing all {len(scores)} responses for challenge {challenge['challenge_id']}")

        # Cleanup
        logger.info("Starting evaluation queue cleanup...")
        await eval_queue.stop_processing()
        logger.info("Evaluation queue cleanup completed, continuing with next iteration...")

    except Exception as e:
        logger.error(f"Error in evaluate_pending_responses: {str(e)}")
        if eval_queue:
            logger.info("Cleaning up evaluation queue after error...")
            await eval_queue.stop_processing()
            logger.info("Cleanup completed after error")

async def run_evaluation_loop(
    db_path: str,
    openai_api_key: str,
    validator_hotkey: str,
    batch_size: int = 10,
    sleep_interval: int = 60
) -> None:
    """Entrypoint that sets up the DB, validator, and runs the loop."""
    try:
        logger.info("Initializing evaluation loop...")
        db_manager = DatabaseManager(db_path)
        validator = GSRValidator(openai_api_key=openai_api_key, validator_hotkey=validator_hotkey)
        validator.db_manager = db_manager  # Let the validator store frame-level evaluations
        iteration = 0
        
        while True:
            try:
                iteration += 1
                logger.info(f"Starting evaluation loop iteration {iteration}")
                logger.info("Getting database connection...")
                
                # Get pending challenges
                conn = db_manager.get_connection()
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                try:
                    logger.info("Checking for challenges ready for evaluation...")
                    # Get challenges ready for evaluation
                    cursor.execute("""
                        SELECT DISTINCT c.challenge_id, c.video_url, c.type AS challenge_type, 
                               COUNT(r.response_id) as pending_count,
                               MIN(r.received_at) as earliest_received
                        FROM responses r
                        JOIN challenges c ON r.challenge_id = c.challenge_id
                        WHERE r.evaluated = FALSE
                          AND datetime(r.received_at) <= datetime('now', '-' || ? || ' minutes')
                        GROUP BY c.challenge_id, c.video_url, c.type
                        LIMIT 1
                    """, (VALIDATION_DELAY.total_seconds() / 60,))
                    
                    challenge = cursor.fetchone()
                    
                except Exception as e:
                    logger.error(f"Database error while checking for challenges: {str(e)}")
                    raise
                finally:
                    logger.info("Closing database cursor and connection...")
                    cursor.close()
                    conn.close()
                
                if not challenge:
                    logger.info(f"No challenges ready for evaluation (iteration {iteration})")
                    logger.info(f"Preparing to sleep for {sleep_interval} seconds...")
                    sleep_start = time.time()
                    await asyncio.sleep(sleep_interval)
                    sleep_duration = time.time() - sleep_start
                    logger.info(f"Waking up after sleeping for {sleep_duration:.1f} seconds (iteration {iteration})")
                    continue
                    
                logger.info(f"Processing challenge {challenge['challenge_id']} with {challenge['pending_count']} responses (iteration {iteration})")
                
                try:
                    # Process the challenge
                    logger.info("Starting evaluate_pending_responses...")
                    await evaluate_pending_responses(
                        validator=validator,
                        db_manager=db_manager,
                        challenge=dict(challenge)
                    )
                    logger.info(f"Successfully completed challenge processing (iteration {iteration})")
                except Exception as e:
                    logger.error(f"Error processing challenge {challenge['challenge_id']} (iteration {iteration}): {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
                    # Continue to next iteration even if this challenge failed
                
                logger.info(f"Completed evaluation iteration {iteration}, preparing to sleep for {sleep_interval} seconds...")
                sleep_start = time.time()
                await asyncio.sleep(sleep_interval)
                sleep_duration = time.time() - sleep_start
                logger.info(f"Waking up after sleeping for {sleep_duration:.1f} seconds (iteration {iteration})")
                
            except Exception as e:
                logger.error(f"Error in evaluation loop iteration {iteration}: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                logger.info(f"Preparing to sleep for {sleep_interval} seconds before retry...")
                sleep_start = time.time()
                await asyncio.sleep(sleep_interval)
                sleep_duration = time.time() - sleep_start
                logger.info(f"Waking up after sleeping for {sleep_duration:.1f} seconds to retry after error")
                continue  # Ensure we continue the loop after any error
                
    except Exception as e:
        logger.error(f"Fatal error in run_evaluation_loop: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise  # Re-raise the exception to trigger the task's error callback

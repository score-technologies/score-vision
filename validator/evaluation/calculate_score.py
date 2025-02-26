from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import json
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import GSRResponse, ValidationResult
from validator.db.operations import DatabaseManager
from validator.config import ALPHA_SCORING_MULTIPLICATOR
import httpx
import math

logger = get_logger(__name__)

async def calculate_score(
    evaluation_results: List[Dict[str, Any]], 
    client: httpx.AsyncClient,
    validator_hotkey: str,
    db_manager: DatabaseManager
) -> Dict[str, Dict[str, float]]:
    """
    Calculate scores for completed tasks and store them with reward data.
    """
    try:
        # Track scores for each response
        response_scores = {}  # Using response_id as key
        
        # Collect all processing times across all responses
        task_processing_times = []
        for result in evaluation_results:
            if 'processing_time' in result:
                task_processing_times.append(result['processing_time'])
        
        # Calculate relative processing times
        min_time = min(task_processing_times) if task_processing_times else 0
        max_time = max(task_processing_times) if task_processing_times else 0
        
        logger.info(f"Processing times: Min time: {min_time}, Max time: {max_time}")
        
        # Process each response's results
        for result in evaluation_results:
            response_id = result['response_id']
            node_id = str(result['node_id'])
            miner_hotkey = result['miner_hotkey']
            processing_time = result['processing_time']
            
            # Calculate quality score
            quality_score = result['validation_result'].score
            
            # Calculate speed score
            speed_score = calculate_speed_score(processing_time, min_time, max_time)
            
            # Get availability score
            availability_score = db_manager.get_availability_score(int(node_id))
            
            # Calculate final score
            final_score = (
                quality_score * 0.6 +
                speed_score * 0.3 +
                availability_score * 0.1
            )
            final_score = final_score**(3*ALPHA_SCORING_MULTIPLICATOR)
            logger.info(f"Final score for response {response_id}: {final_score}")
            
            # Parse response data
            try:
                if isinstance(result['task_returned_data'], dict):
                    response_data = result['task_returned_data']
                elif isinstance(result['task_returned_data'], str):
                    response_data = json.loads(result['task_returned_data'])
                else:
                    response_data = {}
            except (json.JSONDecodeError, TypeError, KeyError):
                logger.warning(f"Could not parse response data for node {node_id}, using empty dict")
                response_data = {}
            
            # Handle timestamps
            started_at = result.get('started_at')
            completed_at = result.get('completed_at')
            
            # Format timestamps
            if started_at:
                if isinstance(started_at, datetime):
                    started_at = started_at.isoformat()
                elif not isinstance(started_at, str):
                    started_at = None
                    
            if completed_at:
                if isinstance(completed_at, datetime):
                    completed_at = completed_at.isoformat()
                elif not isinstance(completed_at, str):
                    completed_at = None
            
            # Store scores for this response
            response_scores[response_id] = {
                'node_id': node_id,
                'miner_hotkey': miner_hotkey,
                'quality_score': quality_score,
                'speed_score': speed_score,
                'availability_score': availability_score,
                'final_score': final_score,
                'processing_time': float(processing_time),
                'validation_result': result['validation_result'],
                'task_returned_data': response_data,
                'started_at': started_at,
                'completed_at': completed_at
            }
        
        return response_scores
        
    except Exception as e:
        logger.error(f"Error in calculate_score: {str(e)}", exc_info=True)
        return {}

def calculate_speed_score(processing_time: float, min_time: float, max_time: float) -> float:
    """Calculate speed score based on processing time using exponential scaling."""
    if max_time == min_time:
        return 1.0  # If all times are the same, give full score
        
    # Normalize time to 0-1 range
    normalized_time = 1 - (processing_time - min_time) / (max_time - min_time)
    
    # Apply exponential scaling to more aggressively reward faster times
    # Using exponential decay with base e
    #exp_score = math.exp(-5 * normalized_time)  # -5 controls steepness of decay
    
    #return max(0.0, min(1.0, exp_score))  # Ensure score stays in 0-1 range
    return max(0.0, min(1.0, normalized_time))

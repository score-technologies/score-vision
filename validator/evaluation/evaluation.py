import os
import json
import base64
import httpx
import tempfile
from typing import Dict, List, Tuple, Optional
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import asyncio
from openai import OpenAI
from fiber.logging_utils import get_logger
from validator.challenge.challenge_types import (
    GSRResponse,
    GSRChallenge,
    ValidationResult
)
from validator.config import FRAMES_TO_VALIDATE
from validator.evaluation.prompts import COUNT_PROMPT, VALIDATION_PROMPT

# Add timeout constants
OPENAI_TIMEOUT = 30.0  # seconds
BATCH_TIMEOUT = 60.0   # seconds
FRAME_TIMEOUT = 180.0  # seconds
MAX_CONCURRENT_CALLS = 3
VLM_RATE_LIMIT = 1  # requests per second
VLM_BATCH_SIZE = 10   # number of images per batch

logger = get_logger(__name__)

# Class IDs
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Colors
COLORS = {
    "player": (0, 255, 0),
    "goalkeeper": (0, 0, 255),
    "referee": (255, 0, 0),
    "ball": (0, 255, 255),
    "keypoint": (255, 0, 255)
}

def optimize_coordinates(coords: List[float]) -> List[float]:
    """Round coordinates to 2 decimals."""
    return [round(float(x), 2) for x in coords]

def filter_keypoints(keypoints: List[List[float]]) -> List[List[float]]:
    """Remove zero-coord keypoints; round others to 2 decimals."""
    return [optimize_coordinates(kp) for kp in keypoints if not (kp[0] == 0 and kp[1] == 0)]

class RateLimiter:
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = datetime.now()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = datetime.now()
            time_passed = (now - self.last_update).total_seconds()
            self.tokens = min(self.rate_limit, self.tokens + time_passed * self.rate_limit)
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate_limit
                await asyncio.sleep(wait_time)
                self.tokens = 1
            
            self.tokens -= 1
            self.last_update = now

class GSRValidator:
    def __init__(self, openai_api_key: str, validator_hotkey: str):
        self.openai_api_key = openai_api_key
        self.validator_hotkey = validator_hotkey
        self.db_manager = None
        self._video_cache = {}
        self.client = OpenAI(api_key=openai_api_key)
        self.chutes_api_key = os.environ.get("CHUTES_API_KEY")
        self.chutes_url = "https://chutes-opengvlab-internvl2-5-78b.chutes.ai/v1/chat/completions"
        self.rate_limiter = RateLimiter(VLM_RATE_LIMIT)

    async def ask_vlm_batch(
        self,
        messages_list: List[List[Dict]],
        max_tokens: int = 1000,
        temperature: float = 0.2,
        model_chutes: str = "OpenGVLab/InternVL2_5-78B",
        model_openai: str = "gpt-4o"
    ) -> List[Optional[str]]:
        """
        Process multiple VLM requests in batches with rate limiting.
        """
        results = []
        for messages in messages_list:
            await self.rate_limiter.acquire()
            try:
                result = await self.ask_vlm(messages, max_tokens, temperature, model_chutes, model_openai)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch VLM call error: {str(e)}")
                results.append(None)
        return results

    async def get_reference_counts_batch(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Get reference counts for multiple frames in batches, including an explicit index
        so we can match responses to the correct frames.
        """
        messages_list = []
        for i, frame in enumerate(frames):
            encoded = self.encode_image(frame)
            messages = [
                {"role": "system", "content": "You are an expert at counting objects in soccer match frames."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Frame index {i}: {COUNT_PROMPT}\n"
                                f"Please return counts as JSON in the form: "
                                f"{{\"index\": {i}, \"player\": X, \"soccer ball\": Y, \"referee\": Z, \"goalkeeper\": W}}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}
                        }
                    ]
                }
            ]
            messages_list.append(messages)

        # Process in batches
        batch_size = VLM_BATCH_SIZE
        results = []
        for i in range(0, len(messages_list), batch_size):
            batch = messages_list[i:i + batch_size]
            batch_results = await self.ask_vlm_batch(batch)
            results.extend(batch_results)

        # Prepare a placeholder list so we can insert by index
        processed_results = [{} for _ in frames]

        # Process results
        for content in results:
            try:
                if not content:
                    continue
                cleaned_content = self.clean_json_response(content)
                counts = json.loads(cleaned_content)
                if not isinstance(counts, dict):
                    continue

                # We expect an "index" field to ensure correct alignment
                idx = counts.pop("index", None)
                if idx is None or not (0 <= idx < len(processed_results)):
                    continue

                # Normalize the keys
                normalized = {}
                for key, value in counts.items():
                    key_lower = key.lower()
                    if isinstance(value, (int, float)):
                        if "ball" in key_lower:
                            normalized["ball"] = int(value)
                        elif "goalkeeper" in key_lower:
                            normalized["goalkeeper"] = int(value)
                        elif "referee" in key_lower:
                            normalized["referee"] = int(value)
                        elif "player" in key_lower:
                            normalized["player"] = int(value)

                processed_results[idx] = normalized

            except Exception as e:
                logger.error(f"Error processing reference count result: {str(e)}")

        return processed_results

    async def ask_vlm(
        self,
        messages: List[Dict],
        max_tokens: int = 1000,
        temperature: float = 0.2,
        model_chutes: str = "OpenGVLab/InternVL2_5-78B",
        model_openai: str = "gpt-4o"
    ) -> Optional[str]:
        """
        Single helper for VLM calls. Attempts Chutes first, then OpenAI fallback.
        Returns the raw content string or None if all fail.
        """
        try:
            async def _call():
                payload = {
                    "model": model_chutes,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                logger.info(f"Chutes API request payload: {json.dumps(payload, indent=2)}")
                
                if self.chutes_api_key:
                    try:
                        async with httpx.AsyncClient() as client:
                            resp = await client.post(
                                self.chutes_url,
                                json=payload,
                                headers={"Authorization": f"Bearer {self.chutes_api_key}"},
                                timeout=30.0
                            )
                            resp_json = resp.json()
                            logger.info(f"Chutes API full response: {json.dumps(resp_json, indent=2)}")
                            
                            if not resp.is_success:
                                error_detail = json.dumps(resp_json, indent=2)
                                logger.error(f"Chutes API failed with status {resp.status_code}: {error_detail}")
                                raise Exception(f"Chutes API error: {resp.status_code} - {error_detail}")
                            
                            if 'choices' not in resp_json or not resp_json['choices']:
                                logger.error(f"Unexpected Chutes API response format: {json.dumps(resp_json, indent=2)}")
                                raise Exception("Unexpected Chutes API response format")
                            
                            return resp_json["choices"][0]["message"]["content"]
                    except httpx.HTTPError as e:
                        logger.error(f"Chutes API HTTP error: {str(e)}")
                    except Exception as e:
                        logger.error(f"Chutes API error: {str(e)}")
                    
                    logger.warning("Falling back to OpenAI")

                try:
                    logger.debug(f"OpenAI request payload: {json.dumps(payload, indent=2)}")
                    openai_resp = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.client.chat.completions.create,
                            model=model_openai,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        ),
                        timeout=OPENAI_TIMEOUT
                    )
                    logger.debug(f"OpenAI full response: {json.dumps(openai_resp.model_dump(), indent=2)}")
                    return openai_resp.choices[0].message.content
                except Exception as e:
                    logger.error(f"OpenAI fallback error: {str(e)}")
                return None

            return await asyncio.wait_for(_call(), timeout=OPENAI_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("VLM call timed out")
        except Exception as e:
            logger.error(f"VLM call error: {str(e)}")
        return None

    def clean_json_response(self, content: str) -> str:
        """Enhanced cleaning of VLM responses."""
        if not content:
            return "[]"
        
        # Remove any markdown formatting
        content = content.replace('```json', '').replace('```', '')
        
        # Remove newlines and extra spaces
        content = ' '.join(content.split())
        
        # If it looks like an array but isn't properly formatted
        if '[' in content and ']' in content:
            try:
                # Extract everything between the first [ and last ]
                array_content = content[content.find('['):content.rfind(']')+1]
                # Try parsing it
                json.loads(array_content)
                return array_content
            except json.JSONDecodeError:
                pass
        
        return content

    async def get_reference_counts(self, frame: np.ndarray) -> Dict:
        """
        Count key entities in the frame.
        Uses a system prompt, fallback handled by ask_vlm.
        """
        encoded = self.encode_image(frame)
        messages = [
            {"role": "system", "content": "You are an expert at counting objects in soccer match frames."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": COUNT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"}
                    }
                ]
            }
        ]
        try:
            content = await self.ask_vlm(messages)
            logger.info(f"VLM Response for reference counts: {content}")
            if not content:
                logger.warning("VLM returned empty content for reference counts")
                return {}

            cleaned_content = self.clean_json_response(content)
            logger.info(f"Cleaned VLM Response: {cleaned_content}")
            
            try:
                counts = json.loads(cleaned_content)
                if not isinstance(counts, dict):
                    logger.warning(f"VLM response is not a dictionary: {counts}")
                    return {}
                    
                # Normalize the keys
                normalized = {}
                for key, value in counts.items():
                    key = key.lower()
                    if isinstance(value, (int, float)):
                        if "ball" in key or "soccer ball" in key:
                            normalized["soccer ball"] = int(value)
                        elif "goalkeeper" in key:
                            normalized["goalkeeper"] = int(value)
                        elif "referee" in key:
                            normalized["referee"] = int(value)
                        elif "player" in key:
                            normalized["player"] = int(value)
                
                logger.info(f"Normalized counts: {normalized}")
                return normalized
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse VLM response as JSON: {cleaned_content}")
                return {}
                
        except Exception as e:
            logger.error(f"Reference count error: {str(e)}")
            return {}

    async def download_video(self, video_url: str) -> Path:
        """
        Download video or return from cache if possible. Handles direct URLs or Google Drive.
        """
        if video_url in self._video_cache:
            cached_path = self._video_cache[video_url]
            if cached_path.exists():
                logger.info(f"Using cached video at: {cached_path}")
                return cached_path
            else:
                del self._video_cache[video_url]

        logger.info(f"Downloading video from: {video_url}")
        if 'drive.google.com' in video_url:
            file_id = None
            if 'id=' in video_url:
                file_id = video_url.split('id=')[1].split('&')[0]
            elif '/d/' in video_url:
                file_id = video_url.split('/d/')[1].split('/')[0]
            if not file_id:
                raise ValueError("Failed to extract Google Drive file ID from URL")
            video_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

        max_retries, retry_delay, timeout = 3, 5, 60.0
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    resp = await client.get(video_url)
                    resp.raise_for_status()
                    temp_dir = Path(tempfile.gettempdir())
                    path = temp_dir / f"video_{datetime.now().timestamp()}.mp4"
                    path.write_bytes(resp.content)
                    if not path.exists() or path.stat().st_size == 0:
                        raise ValueError("Video is empty/missing")

                    cap = cv2.VideoCapture(str(path))
                    if not cap.isOpened():
                        cap.release()
                        path.unlink(missing_ok=True)
                        raise ValueError("Not a valid video")
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    logger.info(f"Video stats: {frame_count} frames, {fps} FPS")
                    self._video_cache[video_url] = path
                    return path
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All download attempts failed: {str(e)}")
                    if 'path' in locals() and path.exists():
                        path.unlink()
                    raise ValueError(f"Failed to download: {str(e)}")

    async def validate_keypoints(self, frame: np.ndarray, keypoints: list, frame_idx: int) -> float:
        """
        Validate keypoints. Attempts Chutes first, then OpenAI fallback.
        Expects a numeric score from 0.0 to 1.0.
        """
        kp_frame = frame.copy()
        for (x, y) in keypoints:
            if x != 0 and y != 0:
                cv2.circle(kp_frame, (int(x), int(y)), 5, COLORS["keypoint"], -1)

        ref_path = Path(__file__).parent / "pitch-keypoints.jpg"
        ref_img = cv2.imread(str(ref_path))

        success_ref, ref_buf = cv2.imencode('.jpg', ref_img)
        success_kp, kp_buf = cv2.imencode('.jpg', kp_frame)
        if not (success_ref and success_kp):
            return 0.0

        ref_b64 = base64.b64encode(ref_buf).decode('utf-8')
        kp_b64 = base64.b64encode(kp_buf).decode('utf-8')

        text_prompt = (
            "Here are two images:\n1) Reference pitch keypoints.\n2) Predicted keypoints.\n"
            "Rate similarity from 0.0 (bad) to 1.0 (perfect). Return only the numeric value."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{kp_b64}"}}
                ]
            }
        ]
        try:
            content = await self.ask_vlm(messages, max_tokens=200)
            if not content:
                return 0.0
            content = self.clean_json_response(content)
            return max(0.0, min(1.0, float(content)))
        except Exception as e:
            logger.error(f"Keypoint validation error: {str(e)}")
            return 0.0

    def resize_frame(self, frame: np.ndarray, target_width: int = 400) -> np.ndarray:
        """Keep aspect ratio on resize."""
        h, w = frame.shape[:2]
        aspect = w / h
        return cv2.resize(frame, (target_width, int(target_width / aspect)))

    async def evaluate_response(
        self,
        response: GSRResponse,
        challenge: GSRChallenge,
        video_path: Path,
        frame_cache: Dict = None,
        frames_to_validate: List[int] = None
    ) -> ValidationResult:
        """
        Main entry to evaluate a GSR response.
        """
        if not getattr(response, 'response_id', None):
            raise ValueError("response_id is required")
        
        node_id = getattr(response, 'node_id', None)
        if node_id is None:
            raise ValueError("node_id is required")

        # Use provided frames or select new ones
        if frames_to_validate is None:
            frames_to_validate = self.select_random_frames(video_path)
        logger.info(f"Evaluating frames: {frames_to_validate}")

        tasks = []
        for frame_idx in frames_to_validate:
            try:
                # Use cached frame and reference counts if available
                if frame_cache and frame_idx in frame_cache:
                    frame = frame_cache[frame_idx]['frame']
                    ref_counts = frame_cache[frame_idx].get('reference_counts')
                else:
                    # Read frame from video
                    cap = cv2.VideoCapture(str(video_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        logger.error(f"Failed to read frame {frame_idx}")
                        continue
                    ref_counts = None

                frame_data = response.frames.get(str(frame_idx), {})
                tasks.append((frame_idx, asyncio.create_task(
                    self.validate_frame_detections(
                        frame=frame,
                        detections=frame_data,
                        frame_idx=frame_idx,
                        challenge_id=challenge.challenge_id,
                        node_id=response.node_id,
                        reference_counts=ref_counts,
                        response_id=response.response_id
                    )
                )))
            except Exception as e:
                logger.error(f"Error preparing frame {frame_idx}: {str(e)}")

        frame_evals = []
        for frame_idx, task in tasks:
            try:
                data = await task
                data["frame_number"] = frame_idx
                frame_evals.append(data)
                # Store in DB
                if self.db_manager:
                    scores = data["scores"]
                    feed = {
                        "scores": {
                            "keypoint_score": scores["keypoint_score"],
                            "bbox_score": scores["bbox_score"],
                            "count_match_score": scores["count_match_score"]
                        },
                        "reference_counts": data["count_validation"]["reference_counts"],
                        "detected_counts": data["count_validation"]["high_confidence_counts"],
                        "count_matches": data["count_validation"]["count_matches"]
                    }
                    self.db_manager.store_frame_evaluation(
                        response_id=response.response_id,
                        challenge_id=challenge.challenge_id,
                        miner_hotkey=response.miner_hotkey,
                        node_id=response.node_id,
                        frame_id=frame_idx,
                        frame_timestamp=frame_idx / 30.0,  # assume 30fps
                        frame_score=scores["final_score"],
                        raw_frame_path="",
                        annotated_frame_path=data["debug_frame_path"],
                        vlm_response=data["count_validation"]["reference_counts"],
                        feedback=json.dumps(feed)
                    )
            except Exception as e:
                logger.error(f"Error evaluating frame {frame_idx}: {str(e)}")

        if not frame_evals:
            return ValidationResult(
                score=0.0,
                frame_scores={},
                feedback="No frames evaluated successfully."
            )

        # Gather results
        total_scores, frame_scores, details = [], {}, []
        for item in frame_evals:
            frm_num = item["frame_number"]
            final_score = item["scores"]["final_score"]
            total_scores.append(final_score)
            frame_scores[frm_num] = final_score
            details.append({
                "frame_number": frm_num,
                "debug_frame_path": item["debug_frame_path"],
                "scores": item["scores"],
                "scoring_details": item["scoring_details"]
            })

        avg_score = sum(total_scores) / len(total_scores) if total_scores else 0.0
        summary = {
            "node_id": response.node_id,
            "challenge_id": challenge.challenge_id,
            "average_score": avg_score,
            "frame_count": len(frame_evals),
            "frame_details": [
                {
                    "frame_number": d["frame_number"],
                    "debug_frame_path": d["debug_frame_path"],
                    "scores": d["scores"]
                } for d in details
            ]
        }
        logger.info(f"Validation Results:\n{json.dumps(summary, indent=2)}")

        return ValidationResult(
            score=avg_score,
            frame_scores=frame_scores,
            feedback=details
        )

    # async def evaluate_frame(
    #     self,
    #     frame_number: int,
    #     frame: np.ndarray,
    #     frame_data: dict,
    #     challenge: GSRChallenge,
    #     response: GSRResponse
    # ) -> float:
    #     """
    #     Evaluate one frame's data. (Kept for reference or modular usage.)
    #     """
    #     ref_counts = await self.get_reference_counts(frame)
    #     filtered = self.filter_detections(frame_data, frame.shape)
    #     bbox_score = await self.evaluate_bboxes(frame, filtered.get("objects", []))
    #     kpt_score = await self.validate_keypoints(frame, filtered.get("keypoints", []), frame_number)
    #     count_val = self.compare_with_reference_counts(ref_counts, {"objects": [], "keypoints": []})
    #     return self.calculate_final_score(kpt_score, bbox_score, count_val["match_score"])

    def filter_detections(self, detections: Dict, shape: Tuple[int, int]) -> Dict:
        """Clamp bboxes, keep valid ones, preserve keypoints."""
        valid = {"objects": [], "keypoints": detections.get("keypoints", [])}
        for obj in detections.get("objects", []):
            bbox = self.validate_bbox_coordinates(obj["bbox"], shape, obj["class_id"])
            if bbox:
                valid["objects"].append({**obj, "bbox": bbox})
        return valid

    async def validate_bbox_content(self, image: np.ndarray, expected_class: str) -> float:
        """
        Validate a single bounding box. Returns probability (0.0..1.0) that
        the cropped image contains the class in question.
        """
        ok, buf = cv2.imencode('.jpg', image)
        if not ok:
            return 0.0
        b64 = base64.b64encode(buf).decode('utf-8')
        prompt = (
            f"This image supposedly has a {expected_class} in a soccer context. "
            "Rate probability [0.0..1.0]. Return only the numeric probability."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }
        ]
        content = await self.ask_vlm(messages, max_tokens=1000)
        if not content:
            return 0.0
        content = self.clean_json_response(content)
        try:
            return max(0.0, min(1.0, float(content)))
        except ValueError:
            return 0.0

    def calculate_bbox_confidence_score(self, results: dict) -> float:
        """
        Weighted average of all object validation scores (0..1).
        Different classes get different weighting.
        """
        objs = results.get("objects", [])
        if not objs:
            return 0.0

        total, weight_sum = 0.0, 0.0
        weights = {
            "soccer ball": 0.7,
            "goalkeeper": 0.3,
            "referee": 0.2,
            "soccer player": 1.0
        }
        for o in objs:
            cls_name = o["class"]
            prob = o["probability"]
            w = weights.get(cls_name, 0.5)
            total += prob * w
            weight_sum += w
        return total / weight_sum if weight_sum else 0.0

    def calculate_final_score(self, keypoint_score: float, bbox_score: float, count_match: float) -> float:
        """
        Combine keypoints, bboxes, and object counts into final 0..1.
        """
        KEY_W, BOX_W, CNT_W = 0.4, 0.4, 0.2
        return (
            (keypoint_score * KEY_W) +
            (bbox_score * BOX_W) +
            (count_match * CNT_W)
        )

    def compare_with_reference_counts(self, ref_counts: Dict[str, int], results: Dict) -> Dict:
        """
        Compare the number of high-confidence detections vs. reference counts.
        Returns normalized match scores per class and overall match score.
        """
        # Normalize reference counts first
        normalized_ref = {
            "player": 0,
            "goalkeeper": 0,
            "referee": 0,
            "ball": 0
        }
        
        if ref_counts:
            for key, value in ref_counts.items():
                key = key.lower()
                if isinstance(value, (int, float)):
                    if "ball" in key or "soccer ball" in key:
                        normalized_ref["ball"] = int(value)
                    elif "goalkeeper" in key:
                        normalized_ref["goalkeeper"] = int(value)
                    elif "referee" in key:
                        normalized_ref["referee"] = int(value)
                    elif "player" in key:
                        normalized_ref["player"] = int(value)

        # Count high confidence detections (prob >= 0.7)
        high_conf = {
            "player": 0,
            "goalkeeper": 0,
            "referee": 0,
            "ball": 0
        }

        for obj in results.get("objects", []):
            if obj.get("probability", 0) >= 0.7:
                cls = obj["class"].lower()
                if "ball" in cls:
                    high_conf["ball"] += 1
                elif "goalkeeper" in cls:
                    high_conf["goalkeeper"] += 1
                elif "referee" in cls:
                    high_conf["referee"] += 1
                elif "player" in cls:
                    high_conf["player"] += 1

        # Calculate match scores with importance weights
        weights = {
            "player": 0.4,    # Players are most important
            "goalkeeper": 0.3, # Goalkeeper is important for team composition
            "referee": 0.2,   # Referee is less critical but should be counted
            "ball": 0.1      # Ball count is least important (usually just 1)
        }

        matches = {}
        weighted_sum = 0
        total_weight = 0

        for cls in ["player", "goalkeeper", "referee", "ball"]:
            ref = normalized_ref[cls]
            detected = high_conf[cls]
            
            # Calculate match score for this class
            if ref == 0 and detected == 0:
                matches[cls] = 1.0  # Perfect match when both are 0
            elif ref == 0:
                matches[cls] = 0.0  # Penalize false positives
            else:
                # Calculate ratio and apply penalties
                ratio = min(detected, ref) / ref
                # Penalize overcounting more than undercounting
                if detected > ref:
                    excess = (detected - ref) / ref
                    penalty = min(1.0, excess * 0.5)  # 50% penalty per excess count
                    ratio = max(0.0, ratio - penalty)
                matches[cls] = max(0.0, min(1.0, ratio))

            weight = weights[cls]
            weighted_sum += matches[cls] * weight
            total_weight += weight

        # Calculate final match score
        match_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Log detailed information for debugging
        logger.info("Count matching details:")
        logger.info(f"Reference counts: {normalized_ref}")
        logger.info(f"Detected counts: {high_conf}")
        logger.info(f"Individual matches: {matches}")
        logger.info(f"Final match score: {match_score}")

        return {
            "reference_counts": normalized_ref,
            "high_confidence_counts": high_conf,
            "count_matches": matches,
            "match_score": match_score
        }

    def select_random_frames(self, video_path: Path, num_frames: int = None) -> List[int]:
        """
        Randomly pick frames from a video, skipping start/end buffer.
        """
        num_frames = num_frames or FRAMES_TO_VALIDATE
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        buffer = min(30, total // 10)
        frange = range(buffer, max(buffer, total - buffer))
        frames = random.sample(frange, min(num_frames, len(frange))) if frange else []
        return sorted(frames)

    def encode_image(self, image):
        """Base64-encode an image."""
        ok, buf = cv2.imencode('.jpg', image)
        return base64.b64encode(buf).decode('utf-8') if ok else ""

    async def validate_bbox_content_batch(self, images: List[np.ndarray], expected: str) -> List[float]:
        """
        Validate multiple bounding boxes in one call with rate limiting.
        """
        if not images:
            return []

        batch_size = min(len(images), VLM_BATCH_SIZE)
        batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
        
        all_probabilities = []
        current_index = 0

        for batch in batches:
            await self.rate_limiter.acquire()
            try:
                # Modified prompt to be more explicit about response format
                content_list = [
                    {
                        "type": "text",
                        "text": (
                            "You are analyzing multiple bounding boxes from a soccer match frame.\n"
                            f"For each box, determine if it contains a {expected}.\n"
                            "CRITICAL: Respond with a single JSON array containing ONLY probability values [0..1].\n"
                            "Example response for 3 boxes: [0.95, 0.82, 0.1]\n"
                            f"Now analyzing {len(batch)} boxes:"
                        )
                    }
                ]
                
                # Add individual box prompts
                for i in range(len(batch)):
                    content_list.append({
                        "type": "text",
                        "text": f"Box {i+1}/{len(batch)}: Determine probability [0..1] that this contains a {expected}."
                    })
                
                messages = [{"role": "user", "content": content_list}]
                
                # Add images
                for img in batch:
                    ok, buf = cv2.imencode('.jpg', img)
                    if not ok:
                        continue
                    b64 = base64.b64encode(buf).decode('utf-8')
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })

                api_response = await self.ask_vlm(messages, max_tokens=100)
                
                if not api_response:
                    all_probabilities.extend([0.0] * len(batch))
                    current_index += len(batch)
                    continue

                # Enhanced parsing logic
                try:
                    # Clean and parse the response
                    cleaned = self.clean_json_response(api_response)
                    
                    # Try parsing as a JSON array first
                    try:
                        probabilities = json.loads(cleaned)
                        if isinstance(probabilities, list):
                            parsed_probs = []
                            for p in probabilities:
                                if isinstance(p, (int, float)):
                                    parsed_probs.append(max(0.0, min(1.0, float(p))))
                                else:
                                    parsed_probs.append(0.0)
                            all_probabilities.extend(parsed_probs)
                            continue
                    except json.JSONDecodeError:
                        pass
                    
                    # Fallback: try to extract numbers from the response
                    import re
                    numbers = re.findall(r'0\.\d+|\d+\.?\d*', cleaned)
                    if numbers:
                        parsed_probs = [max(0.0, min(1.0, float(n))) for n in numbers[:len(batch)]]
                        all_probabilities.extend(parsed_probs)
                    else:
                        all_probabilities.extend([0.0] * len(batch))
                        
                except Exception as e:
                    logger.error(f"Batch parse error: {str(e)}")
                    all_probabilities.extend([0.0] * len(batch))
                    
            except Exception as e:
                logger.error(f"Batch validation error: {str(e)}")
                all_probabilities.extend([0.0] * len(batch))

            current_index += len(batch)
            await asyncio.sleep(0.1)

        # Ensure we return exactly len(images) probabilities
        return all_probabilities[:len(images)]

    async def validate_frame_detections(
        self,
        frame: np.ndarray,
        detections: dict,
        frame_idx: int,
        challenge_id: str,
        node_id: int,
        reference_counts: Dict = None,
        response_id: str = None
    ) -> dict:
        """
        Validate detections in a single frame with timeout protection.
        """
        # Initialize results with default values
        results = {
            "objects": [],
            "keypoints": {"score": 0.0, "points": [], "visualization_path": ""},
            "scores": {
                "keypoint_score": 0.0,
                "bbox_score": 0.0,
                "count_match_score": 0.0,
                "final_score": 0.0
            },
            "debug_frame_path": "",
            "timing": {}
        }

        try:
            async def _process_frame():
                start = datetime.now()
                logger.info(f"Validating frame {frame_idx} (challenge {challenge_id}, node {node_id})")

                # Run validations concurrently
                filtered = self.filter_detections(detections, frame.shape)
                kpts = filtered.get("keypoints", [])
                kpt_score_task = self.validate_keypoints(frame, kpts, frame_idx)

                # Process objects first
                class_map = {}
                for obj in filtered.get("objects", []):
                    cls = self.get_class_name(obj["class_id"])
                    x1, y1, x2, y2 = obj["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    if cls not in class_map:
                        class_map[cls] = {"images": [], "objs": []}
                    class_map[cls]["images"].append(crop)
                    class_map[cls]["objs"].append(obj)

                # Process each class with rate limiting
                obj_scores = []
                for cls, group in class_map.items():
                    confs = await self.validate_bbox_content_batch(group["images"], cls)
                    for ob, c in zip(group["objs"], confs):
                        obj_scores.append({"class": cls, "score": c})
                        results["objects"].append({
                            "bbox_idx": ob["id"],
                            "class": cls,
                            "class_id": ob["class_id"],
                            "probability": c
                        })

                # Calculate scores
                box_score = self.calculate_bbox_confidence_score(results)
                
                # Get reference counts after object validation
                ref_counts = reference_counts or await self.get_reference_counts(frame)
                kpt_score = await kpt_score_task

                count_val = self.compare_with_reference_counts(ref_counts, results)
                final_score = self.calculate_final_score(kpt_score, box_score, count_val["match_score"])

                # Save debug visualization
                ann = self.draw_annotations(frame, filtered)
                ann_resized = self.resize_frame(ann, target_width=400)

                current_date = datetime.now().strftime("%Y%m%d")
                dbg_dir = Path("debug_frames") / current_date
                dbg_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Include response_id in filename if available
                filename_parts = [
                    f"challenge_{challenge_id}",
                    f"node_{node_id}",
                    f"frame_{frame_idx}"
                ]
                if response_id:
                    filename_parts.insert(1, f"resp_{response_id}")  # Insert after challenge_id
                    
                dbg_path = dbg_dir / f"{('_'.join(filename_parts))}_{ts}.jpg"
                cv2.imwrite(str(dbg_path), ann_resized)

                # Update results
                results["scores"].update({
                    "keypoint_score": kpt_score,
                    "bbox_score": box_score,
                    "count_match_score": count_val["match_score"],
                    "final_score": final_score
                })
                results["debug_frame_path"] = str(dbg_path)
                results["count_validation"] = count_val
                results["scoring_details"] = {
                    "keypoint_score": {"score": kpt_score, "weight": 0.4},
                    "bbox_score": {"score": box_score, "weight": 0.4, "object_scores": obj_scores},
                    "count_match_score": {
                        "score": count_val["match_score"],
                        "weight": 0.2,
                        "reference_counts": ref_counts,
                        "detected_counts": count_val["high_confidence_counts"]
                    }
                }
                results["timing"] = {
                    "total_time": (datetime.now() - start).total_seconds()
                }

            await asyncio.wait_for(_process_frame(), timeout=FRAME_TIMEOUT)

        except asyncio.TimeoutError:
            logger.error(f"Frame {frame_idx} validation timed out")
        except Exception as e:
            logger.error(f"Frame {frame_idx} validation error: {str(e)}")

        return results

    def draw_annotations(self, frame: np.ndarray, detections: dict) -> np.ndarray:
        """Draw bounding boxes and keypoints onto the frame."""
        out = frame.copy()
        for obj in detections.get("objects", []):
            (x1, y1, x2, y2) = obj["bbox"]
            cid = obj["class_id"]
            if cid == BALL_CLASS_ID:
                color = COLORS["ball"]
            elif cid == GOALKEEPER_CLASS_ID:
                color = COLORS["goalkeeper"]
            elif cid == REFEREE_CLASS_ID:
                color = COLORS["referee"]
            else:
                color = COLORS["player"]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        for (x, y) in detections.get("keypoints", []):
            if x != 0 and y != 0:
                cv2.circle(out, (int(x), int(y)), 5, COLORS["keypoint"], -1)
        return out

    def get_class_name(self, class_id: int) -> str:
        """Map class_id to string. (Legacy usage retained.)"""
        names = {0: "soccer ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        return names.get(class_id, "unknown")

    def validate_bbox_coordinates(
        self,
        bbox: List[float],
        frame_shape: Tuple[int, int],
        class_id: int
    ) -> Optional[List[int]]:
        """
        Clamp bbox to frame bounds. Discard invalid or tiny ones.
        """
        try:
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            if x2 <= x1 or y2 <= y1:
                return None
            x1, x2 = sorted([max(0, min(x1, w)), max(0, min(x2, w))])
            y1, y2 = sorted([max(0, min(y1, h)), max(0, min(y2, h))])
            if class_id == 0:  # ball can be small
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    return None
            else:
                if (x2 - x1) < 5 or (y2 - y1) < 5:
                    return None
            return [x1, y1, x2, y2]
        except Exception as e:
            logger.error(f"BBox validation error: {str(e)}")
            return None
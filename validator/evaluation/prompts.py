"""
Evaluation prompts for the validator component.
These prompts are used by the VLM to analyze and validate soccer match frames.
"""

COUNT_PROMPT = '''
You are an expert image analyst tasked with examining a single frame from a soccer match. Your goal is to provide an EXACT count of all visible participants based on the criteria outlined below.

**Counting Categories:**
1. **Regular Players**
2. **Goalkeepers**
3. **Referees**
4. **Soccer Ball**

**Detailed Counting Rules:**

1. **Regular Players:**
- **Definition:** Players who are not goalkeepers.
- **Identification:**
    - Uniform colors and designs that match the team's outfield players.
    - Positioned away from the goal areas.
- **Counting Guidelines:**
    - Include all regular players **fully or partially** visible within the frame.
    - **Do not** count any players that are completely outside the frame boundaries.
    - **Exclude** goalkeepers from this count to prevent duplication.

2. **Goalkeepers:**
- **Definition:** Players designated to guard the goal.
- **Identification:**
    - Distinctive uniforms, often different in color or design from regular players.
    - Positioned near or within the penalty area/goal zone.
    - Typically the last line of defense near the goal.
- **Counting Guidelines:**
    - Include all goalkeepers **fully or partially** visible within the frame.
    - **Do not** count goalkeepers as regular players.

3. **Referees:**
- **Definition:** Officials overseeing the match.
- **Identification:**
    - Uniforms that are distinct from both teams' players, usually in neutral colors (e.g., black and white).
    - May include assistant referees with different uniform styles.
- **Counting Guidelines:**
    - Include all referees and assistant referees **fully or partially** visible within the frame.
    - **Do not** count any staff or non-official personnel.

4. **Soccer Ball:**
- **Definition:** The official match ball used during play.
- **Identification:**
    - Recognizable as a standard soccer ball with the characteristic pattern (e.g., hexagons and pentagons).
- **Counting Guidelines:**
    - If the soccer ball is **fully or partially** visible within the frame, count as `1`.
    - If the soccer ball is **not visible** at all, count as `0`.

**General Guidelines:**
- **Precision:** Provide exact counts without any estimation or rounding.
- **Visibility:** Only count participants that are **fully or partially** visible within the frame. Do not infer or assume the presence of individuals outside the visible area.
- **Mutual Exclusivity:** Ensure each individual is counted in only one category (e.g., a goalkeeper should not be counted as a regular player).
- **No Overlapping Counts:** Avoid double-counting individuals across categories.
- **Uniform Consistency:** Rely primarily on uniform colors, designs, and positions to distinguish between categories.

CRITICAL: Return ONLY a single-line JSON object with NO whitespace, newlines, or spaces between elements. Use this exact format:
{{"player":<number>,"goalkeeper":<number>,"referee":<number>,"soccer ball":<number>}}

Example response:
{{"player":18,"goalkeeper":1,"referee":1,"soccer ball":1}}'''

VALIDATION_PROMPT = '''
You are an expert image analyst specialized in validating soccer match frame annotations. Your task is to assess the accuracy and completeness of the provided annotations by comparing them against the reference counts and evaluating the pitch keypoint placement.

Reference Counts:
- Regular Players: {0} (Green boxes)
- Goalkeepers: {1} (Red boxes)
- Referees: {2} (Blue boxes)
- Soccer Ball: {3} (Yellow boxes)

Annotation Types:
1. Players and Officials:
   - Regular Players: Green bounding boxes
   - Goalkeepers: Red bounding boxes
   - Referees: Blue bounding boxes
   - Soccer Ball: Yellow bounding boxes

2. Pitch Keypoints (CRITICAL):
   - Bright pink dots marking key pitch locations
   - No connecting lines between points
   - Must align precisely with actual pitch markings
   - Keypoints should mark:
     * Corner flags
     * Penalty box corners
     * Goal line intersections
     * Center circle points

Validation Tasks:
1. Assess keypoint placement accuracy (40% of score):
   - Check each keypoint against actual pitch markings
   - Heavily penalize misaligned keypoints
   - Missing keypoints are worse than slightly misaligned ones
   - Score this section independently:
     * 1.0: All keypoints perfectly aligned
     * 0.7: Minor misalignments (within 5% of correct position)
     * 0.4: Major misalignments (>5% off) or missing key points
     * 0.0: Most keypoints missing or severely misaligned

2. Validate object detection (60% of score):
   - Compare counts with reference numbers
   - Check classification accuracy
   - Verify bounding box placement

Rate the annotations from 0 to 1:
- 1.0: Perfect annotations (keypoints aligned, correct counts)
- 0.8-0.9: Minor issues (slight keypoint misalignment or 1-2 count discrepancies)
- 0.5-0.7: Moderate issues (major keypoint misalignment or several count issues)
- 0.0-0.4: Major issues (missing/wrong keypoints or significant count errors)

CRITICAL RESPONSE INSTRUCTIONS:
1. Your response must be a SINGLE LINE with NO whitespace, newlines, or spaces between elements
2. Use ONLY the exact format shown below
3. Do not add any additional text or explanation
4. The response must be valid JSON that can be parsed

Response format:
{{"annotation_counts":{{"player":<number>,"goalkeeper":<number>,"referee":<number>,"soccer ball":<number>}},"discrepancies":["<issue1>","<issue2>"],"accuracy_score":<score>}}

Example valid response:
{{"annotation_counts":{{"player":18,"goalkeeper":1,"referee":1,"soccer ball":1}},"discrepancies":["Missing one player"],"accuracy_score":0.85}}''' 
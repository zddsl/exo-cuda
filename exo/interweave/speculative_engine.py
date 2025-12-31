#!/usr/bin/env python3
"""
Speculative Execution Engine

Never waste GPU cycles! When idle:
1. Pattern match - Learn user request patterns
2. Speculative decode - Pre-compute likely continuations
3. Predictive prefetch - Generate probable next responses
4. Warmup tasks - Keep GPU hot with useful work

If speculation is wrong, discard. If right, instant response!
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
import hashlib
import json


@dataclass
class RequestPattern:
    """Learned pattern from user requests"""
    prompt_hash: str
    prompt_prefix: str  # First N tokens
    frequency: int = 1
    last_seen: float = field(default_factory=time.time)
    avg_response_tokens: int = 50
    common_followups: List[str] = field(default_factory=list)


@dataclass
class SpeculativeResult:
    """Pre-computed speculative result"""
    prompt_hash: str
    tokens: List[int]
    logits: Optional[np.ndarray] = None
    confidence: float = 0.0
    computed_at: float = field(default_factory=time.time)
    hit_count: int = 0


class PatternMatcher:
    """
    Learn user request patterns to predict future requests.

    Tracks:
    - Common prompt prefixes
    - Frequent follow-up patterns
    - Time-of-day patterns
    - Conversation flow patterns
    """

    def __init__(self, max_patterns: int = 1000):
        self.patterns: Dict[str, RequestPattern] = {}
        self.recent_prompts: Deque[str] = deque(maxlen=100)
        self.max_patterns = max_patterns

        # Common patterns to pre-learn
        self.seed_patterns = [
            "Hello",
            "What is",
            "How do",
            "Can you",
            "Explain",
            "Write a",
            "Help me",
            "Tell me",
        ]

    def _hash_prompt(self, prompt: str) -> str:
        """Create hash of prompt prefix"""
        prefix = prompt[:50] if len(prompt) > 50 else prompt
        return hashlib.md5(prefix.encode()).hexdigest()[:16]

    def record_request(self, prompt: str, response_tokens: int = 0):
        """Record a user request to learn patterns"""
        prompt_hash = self._hash_prompt(prompt)
        prefix = prompt[:100]

        if prompt_hash in self.patterns:
            pattern = self.patterns[prompt_hash]
            pattern.frequency += 1
            pattern.last_seen = time.time()
            if response_tokens > 0:
                # Running average
                pattern.avg_response_tokens = (
                    pattern.avg_response_tokens * 0.9 + response_tokens * 0.1
                )
        else:
            self.patterns[prompt_hash] = RequestPattern(
                prompt_hash=prompt_hash,
                prompt_prefix=prefix,
                avg_response_tokens=response_tokens or 50,
            )

        # Track follow-up patterns
        if len(self.recent_prompts) > 0:
            prev_hash = self._hash_prompt(self.recent_prompts[-1])
            if prev_hash in self.patterns:
                followups = self.patterns[prev_hash].common_followups
                if prompt_hash not in followups:
                    followups.append(prompt_hash)
                    # Keep only top 5 follow-ups
                    if len(followups) > 5:
                        followups.pop(0)

        self.recent_prompts.append(prompt)

        # Prune old patterns
        if len(self.patterns) > self.max_patterns:
            self._prune_patterns()

    def _prune_patterns(self):
        """Remove least used patterns"""
        # Sort by frequency * recency
        scored = [
            (h, p.frequency * (1.0 / (time.time() - p.last_seen + 1)))
            for h, p in self.patterns.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Keep top half
        keep = set(h for h, _ in scored[:self.max_patterns // 2])
        self.patterns = {h: p for h, p in self.patterns.items() if h in keep}

    def get_likely_prompts(self, n: int = 5) -> List[RequestPattern]:
        """Get most likely next prompts based on patterns"""
        # Weight by frequency and recency
        now = time.time()
        scored = []
        for pattern in self.patterns.values():
            recency = 1.0 / (now - pattern.last_seen + 1)
            score = pattern.frequency * recency
            scored.append((score, pattern))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:n]]

    def get_followup_predictions(self, current_prompt: str) -> List[RequestPattern]:
        """Predict likely follow-up requests"""
        prompt_hash = self._hash_prompt(current_prompt)

        if prompt_hash not in self.patterns:
            return []

        pattern = self.patterns[prompt_hash]
        predictions = []

        for followup_hash in pattern.common_followups:
            if followup_hash in self.patterns:
                predictions.append(self.patterns[followup_hash])

        return predictions


class SpeculativeCache:
    """
    Cache of pre-computed speculative results.

    When GPU is idle, we speculate on likely requests.
    If user makes that request, instant response!
    """

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, SpeculativeResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def store(self, prompt_hash: str, tokens: List[int],
              logits: Optional[np.ndarray] = None, confidence: float = 0.5):
        """Store speculative result"""
        self.cache[prompt_hash] = SpeculativeResult(
            prompt_hash=prompt_hash,
            tokens=tokens,
            logits=logits,
            confidence=confidence,
        )

        # Prune if too large
        if len(self.cache) > self.max_size:
            self._prune()

    def lookup(self, prompt_hash: str) -> Optional[SpeculativeResult]:
        """Check if we have a speculative result"""
        if prompt_hash in self.cache:
            result = self.cache[prompt_hash]
            result.hit_count += 1
            self.hits += 1
            return result
        self.misses += 1
        return None

    def _prune(self):
        """Remove least confident / oldest results"""
        scored = [
            (h, r.confidence * r.hit_count / (time.time() - r.computed_at + 1))
            for h, r in self.cache.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        keep = set(h for h, _ in scored[:self.max_size // 2])
        self.cache = {h: r for h, r in self.cache.items() if h in keep}

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class SpeculativeEngine:
    """
    Main speculative execution engine.

    Runs in background, using idle GPU cycles to:
    1. Learn patterns from user requests
    2. Pre-compute likely next responses
    3. Cache results for instant serving
    """

    def __init__(self, inference_engine, shard):
        self.engine = inference_engine
        self.shard = shard
        self.pattern_matcher = PatternMatcher()
        self.cache = SpeculativeCache()

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self.speculations_run = 0
        self.speculation_time_ms = 0

    async def start(self):
        """Start speculative execution loop"""
        self._running = True
        self._task = asyncio.create_task(self._speculation_loop())
        print("[Speculative] Engine started")

    async def stop(self):
        """Stop speculative execution"""
        self._running = False
        if self._task:
            self._task.cancel()

    async def _speculation_loop(self):
        """Main speculation loop - runs when idle"""
        while self._running:
            try:
                # Get likely prompts
                likely = self.pattern_matcher.get_likely_prompts(3)

                for pattern in likely:
                    # Check if already cached
                    if self.cache.lookup(pattern.prompt_hash):
                        continue

                    # Run speculative inference
                    start = time.perf_counter()
                    try:
                        # Create dummy input from pattern
                        # In real impl, would tokenize prompt_prefix
                        dummy_tokens = np.array([[128000]], dtype=np.int64)

                        result = await self.engine.infer_tensor(
                            f"spec-{pattern.prompt_hash}",
                            self.shard,
                            dummy_tokens
                        )

                        if isinstance(result, tuple):
                            output = result[0]
                        else:
                            output = result

                        # Get predicted token
                        if len(output.shape) == 3:
                            logits = output[0, -1, :]
                        else:
                            logits = output[-1, :]

                        next_token = int(np.argmax(logits))

                        # Cache result
                        self.cache.store(
                            pattern.prompt_hash,
                            tokens=[next_token],
                            logits=logits,
                            confidence=pattern.frequency / 100.0
                        )

                        elapsed = (time.perf_counter() - start) * 1000
                        self.speculations_run += 1
                        self.speculation_time_ms += elapsed

                        print(f"[Speculative] Pre-computed for '{pattern.prompt_prefix[:30]}...' ({elapsed:.1f}ms)")

                    except Exception as e:
                        print(f"[Speculative] Error: {e}")

                # Wait before next speculation round
                await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Speculative] Loop error: {e}")
                await asyncio.sleep(10)

    def record_request(self, prompt: str, response_tokens: int = 0):
        """Record user request for pattern learning"""
        self.pattern_matcher.record_request(prompt, response_tokens)

    def check_cache(self, prompt: str) -> Optional[SpeculativeResult]:
        """Check if we have a pre-computed result"""
        prompt_hash = self.pattern_matcher._hash_prompt(prompt)
        return self.cache.lookup(prompt_hash)

    def get_stats(self) -> dict:
        return {
            'patterns_learned': len(self.pattern_matcher.patterns),
            'cache_size': len(self.cache.cache),
            'cache_hit_rate': f"{self.cache.hit_rate * 100:.1f}%",
            'speculations_run': self.speculations_run,
            'avg_speculation_ms': self.speculation_time_ms / max(1, self.speculations_run),
        }


# ============================================================================
# Conversation Flow Predictor
# ============================================================================

class ConversationPredictor:
    """
    Predict conversation flow to pre-compute responses.

    Common patterns:
    - "What is X?" → "Tell me more about X"
    - "How do I X?" → "Can you give an example?"
    - "Write X" → "Can you modify it to Y?"
    """

    FLOW_PATTERNS = [
        # (trigger_pattern, likely_followups)
        ("what is", ["tell me more", "can you explain", "give an example"]),
        ("how do", ["show me an example", "what if", "can you explain"]),
        ("write a", ["can you modify", "add", "change"]),
        ("explain", ["give an example", "what about", "how does"]),
        ("help me", ["can you also", "what about", "next step"]),
    ]

    def __init__(self):
        self.conversation_history: Deque[str] = deque(maxlen=10)

    def predict_next(self, current_prompt: str) -> List[str]:
        """Predict likely next prompts based on current"""
        predictions = []
        prompt_lower = current_prompt.lower()

        for trigger, followups in self.FLOW_PATTERNS:
            if trigger in prompt_lower:
                predictions.extend(followups)

        return predictions[:3]  # Top 3 predictions

    def record(self, prompt: str):
        """Record prompt for flow analysis"""
        self.conversation_history.append(prompt)


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    # Test pattern matcher
    pm = PatternMatcher()

    # Simulate user requests
    requests = [
        "What is machine learning?",
        "How do neural networks work?",
        "What is machine learning?",  # Repeat
        "Explain backpropagation",
        "What is machine learning?",  # Repeat again
        "How do I train a model?",
    ]

    for req in requests:
        pm.record_request(req)

    print("Learned patterns:")
    for pattern in pm.get_likely_prompts(5):
        print(f"  [{pattern.frequency}x] {pattern.prompt_prefix[:50]}...")

    print("\nFollow-up predictions for 'What is machine learning?':")
    for pred in pm.get_followup_predictions("What is machine learning?"):
        print(f"  → {pred.prompt_prefix[:50]}...")

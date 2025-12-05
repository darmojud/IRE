import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

import json
import requests
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
import pickle
from datetime import datetime
from elasticsearch import Elasticsearch, ElasticsearchWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from scipy import stats
import hashlib

warnings.filterwarnings("ignore", category=ElasticsearchWarning)


class NewsRankingSystem:
    def __init__(self, api_url="http://localhost:3000", es_host="localhost:9200"):
        self.api_url = api_url
        self.es = (
            Elasticsearch("http://localhost:9200", basic_auth=("elastic", "G7r40vJX"))
            if es_host
            else None
        )
        self.logs = []
        self.user_histories = defaultdict(list)
        self.articles = []

        # Models
        self.ltr_weights = None
        self.xgb_model = None
        self.scaler = StandardScaler()

        # Collaborative filtering
        self.user_item_matrix = None
        self.svd_model = None
        self.cf_n_components = 50
        self.user_id_map = {}
        self.article_id_map = {}
        self.reverse_article_map = {}

        # Enhanced experiment tracking with individual metrics
        self.experiment_data = []
        self.variant_metrics = defaultdict(
            lambda: {
                "clicks": [],
                "dwell_times": [],
                "likes": [],
                "shares": [],
                "bookmarks": [],
                "weighted_clicks": [],
                "users": set(),
                "queries": 0,
                "total_impressions": 0,
                "actions_per_query": [],
            }
        )

        # A/B test phase control
        self.ab_test_phase = "baseline"  # "baseline" or "experiment"

        # Monitoring configuration
        self.monitoring_interval = 50  # Log progress every N queries
        self.enable_early_stopping = True
        self.early_stop_min_samples = 30
        self.early_stop_confidence = 0.99

    def save_logs_to_csv(self, filepath="interaction_logs.csv"):
        if not self.logs:
            print("No logs to save")
            return

        rows = []
        for log in self.logs:
            rows.append(
                {
                    "query_id": log.get("query_id"),
                    "user_id": log.get("user_id"),
                    "query_text": log.get("query_text"),
                    "article_id": log.get("article_id"),
                    "position": log.get("position"),
                    "reward": log.get("reward"),
                    "clicked": log.get("clicked"),
                    "timestamp": log.get("timestamp"),
                    "variant": log.get("variant"),
                    "ranking_method": log.get("ranking_method"),
                    "topics": "|".join(log.get("topics", [])),
                    "actions": str(log.get("actions", [])),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(rows)} logs to {filepath}")

    def load_articles(self, filepath="articles.jsonl"):
        self.articles = []
        with open(filepath, "r") as f:
            for line in f:
                article = json.loads(line)
                self.articles.append(
                    {
                        "article_id": article["uuid"],
                        "text": article["text"],
                        "topics": article["topics"],
                    }
                )
        print(f"Loaded {len(self.articles)} articles")
        return self.articles

    def index_articles_to_es(self):
        if not self.es:
            print("Elasticsearch not configured")
            return

        index_name = "news_articles"

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)

        mapping = {
            "mappings": {
                "properties": {
                    "article_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "topics": {"type": "keyword"},
                }
            }
        }
        self.es.indices.create(index=index_name, body=mapping)

        print("Indexing articles to Elasticsearch...")

        for article in self.articles:
            self.es.index(index=index_name, id=article["article_id"], document=article)

        print(f"Indexed {len(self.articles)} articles to Elasticsearch")

    def get_query(self):
        response = requests.get(f"{self.api_url}/query")
        return response.json()

    def submit_ranklist(self, query_id, user_id, ranked_article_ids):
        payload = {
            "query_id": query_id,
            "user_id": user_id,
            "ranked_article_ids": ranked_article_ids,
        }
        response = requests.post(f"{self.api_url}/ranklist", json=payload)
        return response.json()

    def elasticsearch_rank(self, query_text, top_k=20):
        if self.es:
            results = self.es.search(
                index="news_articles",
                body={
                    "query": {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["text^2", "topics"],
                        }
                    },
                    "size": top_k,
                },
            )
            return [hit["_id"] for hit in results["hits"]["hits"]]
        else:
            scored_articles = []
            query_terms = query_text.lower().split()

            for article in self.articles:
                text = (
                    f"{article['text']} {' '.join(article.get('topics', []))}".lower()
                )
                score = sum(text.count(term) for term in query_terms)
                scored_articles.append((article["article_id"], score))

            scored_articles.sort(key=lambda x: x[1], reverse=True)
            return [aid for aid, _ in scored_articles[:top_k]]

    def calculate_propensity(self, position):
        return 1.0 / np.log2(position + 2)

    def calculate_reward(self, actions):
        if not actions:
            return 0.0

        reward = 0.0
        for action in actions:
            if action == "Click":
                reward += 1.0
            elif action == "Like":
                reward += 3.0
            elif action == "Share":
                reward += 5.0
            elif action == "Bookmark":
                reward += 4.0
            elif isinstance(action, dict) and "Dwell" in action:
                dwell_secs = action["Dwell"].get("secs", 0)
                dwell_nanos = action["Dwell"].get("nanos", 0)
                total_secs = dwell_secs + dwell_nanos / 1e9
                dwell_reward = min(np.log1p(total_secs) * 1.5, 3.0)
                reward += dwell_reward

        return reward

    def calculate_engagement_metrics(self, actions):
        metrics = {
            "clicks": 0,
            "dwell_time": 0.0,
            "likes": 0,
            "shares": 0,
            "bookmarks": 0,
            "total_actions": 0,
        }

        for action_list in actions:
            if not action_list:
                continue

            for action in action_list:
                metrics["total_actions"] += 1

                if action == "Click":
                    metrics["clicks"] += 1

                elif action == "Like":
                    metrics["likes"] += 1

                elif action == "Share":
                    metrics["shares"] += 1

                elif action == "Bookmark":
                    metrics["bookmarks"] += 1

                elif isinstance(action, dict) and "Dwell" in action:
                    dwell_secs = action["Dwell"].get("secs", 0)
                    dwell_nanos = action["Dwell"].get("nanos", 0)
                    total_secs = dwell_secs + dwell_nanos / 1e9
                    metrics["dwell_time"] += total_secs

        return metrics

    def calculate_weighted_clicks(self, actions, top_k=10):
        weighted_score = 0.0

        for position, action_list in enumerate(actions[:top_k]):
            if not action_list:
                continue

            position_weight = 1.0 / np.log2(position + 2)
            reward = self.calculate_reward(action_list)
            weighted_score += reward * position_weight

        return weighted_score

    def extract_features(self, article_id, query_text, user_id, position=None):
        query_terms = set(query_text.lower().split())
        article_obj = next(
            (a for a in self.articles if a["article_id"] == article_id), None
        )

        if not article_obj:
            return np.zeros(14)

        text = article_obj.get("text", "").lower()
        topics = article_obj.get("topics", [])

        # Text relevance features
        text_match = sum(1 for term in query_terms if term in text) / max(
            len(query_terms), 1
        )
        topic_text = " ".join(topics).lower()
        topic_match_query = sum(1 for term in query_terms if term in topic_text) / max(
            len(query_terms), 1
        )

        # User preference features
        user_history = self.user_histories.get(user_id, [])
        user_topic_prefs = defaultdict(float)

        for log in user_history[-50:]:
            if log.get("reward", 0) > 0:
                for topic in log.get("topics", []):
                    user_topic_prefs[topic] += log["reward"]

        topic_preference_score = 0.0
        if user_topic_prefs:
            topic_preference_score = sum(
                user_topic_prefs.get(topic, 0) for topic in topics
            )
            topic_preference_score /= sum(user_topic_prefs.values())

        # Content features
        num_topics = len(topics)
        text_length = len(text) / 1000.0

        # Historical performance
        article_clicks = sum(
            1
            for log in self.logs
            if log.get("article_id") == article_id and log.get("clicked", False)
        )
        article_views = sum(
            1 for log in self.logs if log.get("article_id") == article_id
        )
        article_ctr = article_clicks / max(article_views, 1)

        avg_reward = (
            np.mean(
                [
                    log.get("reward", 0)
                    for log in self.logs
                    if log.get("article_id") == article_id
                ]
            )
            if article_views > 0
            else 0
        )

        # User activity
        user_activity = len(user_history) / 100.0
        user_avg_reward = (
            np.mean([log.get("reward", 0) for log in user_history])
            if user_history
            else 0
        )

        # Collaborative filtering score
        cf_score = self.get_cf_score(user_id, article_id)

        features = np.array(
            [
                text_match,
                topic_match_query,
                topic_preference_score,
                num_topics,
                text_length,
                article_ctr,
                avg_reward,
                user_activity,
                user_avg_reward,
                cf_score,
                np.log1p(article_views),
                np.log1p(article_clicks),
                len(user_history) > 0,
                1.0,  # Bias term
            ]
        )

        return features

    def get_cf_score(self, user_id, article_id):
        if (
            self.svd_model is None
            or user_id not in self.user_id_map
            or article_id not in self.article_id_map
        ):
            return 0.0

        try:
            user_idx = self.user_id_map[user_id]
            article_idx = self.article_id_map[article_id]

            user_factors = self.svd_model.components_[:, user_idx]
            item_factors = self.svd_model.components_[:, article_idx]

            score = np.dot(user_factors, item_factors)
            return float(score)
        except:
            return 0.0

    def train_collaborative_filtering(self):
        print("Training collaborative filtering model...")

        if len(self.logs) < 20:
            print("Not enough data for CF")
            return

        users = list(set(log["user_id"] for log in self.logs))
        articles = list(set(log["article_id"] for log in self.logs))

        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.article_id_map = {aid: idx for idx, aid in enumerate(articles)}
        self.reverse_article_map = {
            idx: aid for aid, idx in self.article_id_map.items()
        }

        rows, cols, data = [], [], []
        for log in self.logs:
            user_idx = self.user_id_map[log["user_id"]]
            article_idx = self.article_id_map[log["article_id"]]
            reward = log.get("reward", 0)

            rows.append(user_idx)
            cols.append(article_idx)
            data.append(reward)

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(users), len(articles))
        )

        n_components = min(self.cf_n_components, min(self.user_item_matrix.shape) - 1)
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd_model.fit(self.user_item_matrix.T)

        print(f"CF model trained with {len(users)} users and {len(articles)} articles")

    def train_ltr_model(self, propensity_clip=0.1, learning_rate=0.01, iterations=100):
        print(f"Training LTR model with {len(self.logs)} samples...")

        if len(self.logs) < 10:
            print("Not enough data to train LTR")
            return

        self.ltr_weights = np.random.randn(14) * 0.01

        for iteration in range(iterations):
            gradients = np.zeros(14)
            total_loss = 0
            count = 0

            for log in self.logs:
                features = log["features"]
                reward = log["reward"]
                position = log["position"]

                propensity = max(self.calculate_propensity(position), propensity_clip)
                ips_weight = reward / propensity

                prediction = np.dot(features, self.ltr_weights)
                error = ips_weight - prediction
                gradients += error * features
                total_loss += error**2
                count += 1

            if count > 0:
                self.ltr_weights += learning_rate * gradients / count

                if (iteration + 1) % 20 == 0:
                    avg_loss = total_loss / count
                    print(
                        f"Iteration {iteration + 1}/{iterations}, Loss: {avg_loss:.4f}"
                    )

    def ltr_rank(self, query_text, user_id, top_k=20):
        if self.ltr_weights is None:
            return self.elasticsearch_rank(query_text, top_k)

        scored_articles = []

        for article in self.articles:
            features = self.extract_features(
                article["article_id"], query_text, user_id, position=None
            )
            score = np.dot(features, self.ltr_weights)
            scored_articles.append((article["article_id"], score))

        scored_articles.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored_articles[:top_k]]

    def train_xgboost_model(self, propensity_clip=0.1):
        print(f"Training XGBoost model with {len(self.logs)} samples...")

        if len(self.logs) < 20:
            print("Not enough data to train XGBoost")
            return

        X = np.array([log["features"] for log in self.logs])
        y = np.array([log["reward"] for log in self.logs])

        weights = []
        for log in self.logs:
            propensity = max(
                self.calculate_propensity(log["position"]), propensity_clip
            )
            weights.append(1.0 / propensity)
        weights = np.array(weights)

        dtrain = xgb.DMatrix(X, label=y, weight=weights)

        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }

        self.xgb_model = xgb.train(params, dtrain, num_boost_round=100)
        print("XGBoost model trained")

    def xgboost_rank(self, query_text, user_id, top_k=20):
        if self.xgb_model is None:
            return self.elasticsearch_rank(query_text, top_k)

        scored_articles = []

        for article in self.articles:
            features = self.extract_features(
                article["article_id"], query_text, user_id, position=None
            )
            features_matrix = xgb.DMatrix(features.reshape(1, -1))
            score = self.xgb_model.predict(features_matrix)[0]
            scored_articles.append((article["article_id"], score))

        scored_articles.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored_articles[:top_k]]

    def collaborative_filtering_rank(self, user_id, query_text, top_k=20):
        if self.svd_model is None or user_id not in self.user_id_map:
            return self.elasticsearch_rank(query_text, top_k)

        candidates = self.elasticsearch_rank(query_text, top_k * 2)

        scored_articles = []
        for article_id in candidates:
            cf_score = self.get_cf_score(user_id, article_id)
            scored_articles.append((article_id, cf_score))

        scored_articles.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored_articles[:top_k]]

    def get_variant_for_user(self, user_id):
        if self.ab_test_phase == "baseline":
            return "baseline"

        hash_val = int(hashlib.md5(str(user_id).encode()).hexdigest()[:8], 16)
        variant_idx = hash_val % 4
        variants = ["baseline", "ltr", "xgboost", "collaborative"]
        return variants[variant_idx]

    def get_ranking_by_variant_with_rerank(
        self, variant, query_text, user_id, top_k=20
    ):

        # Stage 1: Initial ranking
        if variant == "ltr" and self.ltr_weights is not None:
            initial_ranked = self.ltr_rank(query_text, user_id, top_k)
            ranking_method = "ltr"
        elif variant == "xgboost" and self.xgb_model is not None:
            initial_ranked = self.xgboost_rank(query_text, user_id, top_k)
            ranking_method = "xgboost"
        elif variant == "collaborative" and self.svd_model is not None:
            initial_ranked = self.collaborative_filtering_rank(
                user_id, query_text, top_k
            )
            ranking_method = "collaborative"
        else:
            initial_ranked = self.elasticsearch_rank(query_text, top_k)
            ranking_method = "baseline"

        # Stage 2: Re-extract features with actual positions
        articles_with_features = []
        for position, article_id in enumerate(initial_ranked):
            features = self.extract_features(
                article_id, query_text, user_id, position=position
            )
            articles_with_features.append(
                {"article_id": article_id, "position": position, "features": features}
            )

        return initial_ranked, ranking_method, articles_with_features

    def monitor_ab_test_progress(self, queries_processed):
        # print(f"\n{'='*90}")
        # print(f"A/B TEST MONITORING - Query {queries_processed}")
        # print(f"{'='*90}")

        for variant in sorted(self.variant_metrics.keys()):
            metrics = self.variant_metrics[variant]

            if not metrics["queries"]:
                continue

            # Calculate rates and averages
            ctr = (
                sum(metrics["clicks"]) / metrics["total_impressions"] * 100
                if metrics["total_impressions"] > 0
                else 0
            )

            avg_dwell = np.mean(metrics["dwell_times"]) if metrics["dwell_times"] else 0

            likes_per_query = (
                sum(metrics["likes"]) / metrics["queries"]
                if metrics["queries"] > 0
                else 0
            )

            shares_per_query = (
                sum(metrics["shares"]) / metrics["queries"]
                if metrics["queries"] > 0
                else 0
            )

            bookmarks_per_query = (
                sum(metrics["bookmarks"]) / metrics["queries"]
                if metrics["queries"] > 0
                else 0
            )

            weighted_clicks_mean = (
                np.mean(metrics["weighted_clicks"]) if metrics["weighted_clicks"] else 0
            )

            actions_per_query = (
                np.mean(metrics["actions_per_query"])
                if metrics["actions_per_query"]
                else 0
            )

            print(f"{variant.upper()}")
            print(f"  Queries: {metrics['queries']:>6}")
            print(f"  Users: {len(metrics['users']):>6}")
            print(f"  Total Impressions: {metrics['total_impressions']:>6}")
            print(f"  ────────────────────────────────────")
            print(f"  CTR:                {ctr:>8.2f}%")
            print(f"  Avg Dwell Time:     {avg_dwell:>8.2f}s")
            print(f"  Likes/Query:        {likes_per_query:>8.3f}")
            print(f"  Shares/Query:       {shares_per_query:>8.3f}")
            print(f"  Bookmarks/Query:    {bookmarks_per_query:>8.3f}")
            print(f"  Actions/Query:      {actions_per_query:>8.2f}")
            print(f"  Weighted Clicks:    {weighted_clicks_mean:>8.4f}")

        print(f"\n{'='*90}\n")

    def should_stop_early(self):
        if not self.enable_early_stopping:
            return False

        baseline_data = self.variant_metrics["baseline"]["weighted_clicks"]

        if len(baseline_data) < self.early_stop_min_samples:
            return False

        for variant in ["ltr", "xgboost", "collaborative"]:
            variant_data = self.variant_metrics[variant]["weighted_clicks"]

            if len(variant_data) < self.early_stop_min_samples:
                continue

            # Run statistical test
            t_stat, p_value = stats.ttest_ind(variant_data, baseline_data)

            variant_mean = np.mean(variant_data)
            baseline_mean = np.mean(baseline_data)
            improvement = (
                ((variant_mean - baseline_mean) / baseline_mean) * 100
                if baseline_mean > 0
                else 0
            )

            # Stop if clear winner (99% confidence + >15% improvement)
            if p_value < (1 - self.early_stop_confidence) and improvement > 15:
                print(f"\n EARLY STOPPING: Clear winner detected!")
                print(f"   Variant: {variant.upper()}")
                print(f"   Improvement: {improvement:.2f}%")
                print(f"   p-value: {p_value:.6f}")
                print(f"   Confidence: {(1-p_value)*100:.2f}%")
                return True

            # Stop if causing significant harm (95% confidence + <-10% degradation)
            if p_value < 0.05 and improvement < -10:
                print(f"\n  EARLY STOPPING: Significant degradation detected!")
                print(f"   Variant: {variant.upper()}")
                print(f"   Degradation: {improvement:.2f}%")
                print(f"   p-value: {p_value:.6f}")
                print(f"   Stopping to prevent harm")
                return True

        return False

    def collect_data_with_ab_testing(self, num_queries=500):

        print(f"\n{'='*90}")
        print(f"COLLECTING {num_queries} QUERIES (Phase: {self.ab_test_phase.upper()})")
        print(f"{'='*90}")

        variant_counts = defaultdict(int)

        for i in range(num_queries):
            try:
                # Get query
                query_data = self.get_query()
                query_id = query_data["query_id"]
                user_id = query_data["user_id"]
                query_text = query_data["query_text"]

                # Get variant assignment
                variant = self.get_variant_for_user(user_id)
                variant_counts[variant] += 1

                # Get ranked list
                ranked_ids, ranking_method, articles_with_features = (
                    self.get_ranking_by_variant_with_rerank(
                        variant, query_text, user_id
                    )
                )

                # Submit and get actions
                result = self.submit_ranklist(query_id, user_id, ranked_ids)
                actions = result["actions"]

                # Calculate all engagement metrics
                engagement_metrics = self.calculate_engagement_metrics(actions)
                weighted_clicks = self.calculate_weighted_clicks(actions)

                # Track metrics by variant
                self.variant_metrics[variant]["clicks"].append(
                    engagement_metrics["clicks"]
                )
                self.variant_metrics[variant]["dwell_times"].append(
                    engagement_metrics["dwell_time"]
                )
                self.variant_metrics[variant]["likes"].append(
                    engagement_metrics["likes"]
                )
                self.variant_metrics[variant]["shares"].append(
                    engagement_metrics["shares"]
                )
                self.variant_metrics[variant]["bookmarks"].append(
                    engagement_metrics["bookmarks"]
                )
                self.variant_metrics[variant]["weighted_clicks"].append(weighted_clicks)
                self.variant_metrics[variant]["actions_per_query"].append(
                    engagement_metrics["total_actions"]
                )
                self.variant_metrics[variant]["users"].add(user_id)
                self.variant_metrics[variant]["queries"] += 1
                self.variant_metrics[variant]["total_impressions"] += len(actions)

                # Log individual interactions for training
                for item, action_list in zip(articles_with_features, actions):
                    article_id = item["article_id"]
                    position = item["position"]
                    features = item["features"]

                    reward = self.calculate_reward(action_list)

                    article_obj = next(
                        (a for a in self.articles if a["article_id"] == article_id),
                        None,
                    )
                    topics = article_obj.get("topics", []) if article_obj else []

                    log_entry = {
                        "query_id": query_id,
                        "user_id": user_id,
                        "query_text": query_text,
                        "article_id": article_id,
                        "position": position,
                        "features": features,
                        "actions": action_list,
                        "reward": reward,
                        "clicked": "Click" in action_list if action_list else False,
                        "topics": topics,
                        "timestamp": datetime.now().isoformat(),
                        "variant": variant,
                        "ranking_method": ranking_method,
                    }

                    self.logs.append(log_entry)
                    self.user_histories[user_id].append(log_entry)

                # Real-time monitoring at intervals
                if (i + 1) % self.monitoring_interval == 0:
                    self.monitor_ab_test_progress(i + 1)

                    print("\nVariant Distribution:")
                    for v in sorted(variant_counts.keys()):
                        pct = (variant_counts[v] / (i + 1)) * 100
                        print(f"  {v:>15}: {variant_counts[v]:>4} ({pct:>5.1f}%)")

                    # Check for early stopping
                    if self.should_stop_early():
                        print(f"\n⚠️  Stopping at query {i + 1}/{num_queries}")
                        break

            except Exception as e:
                print(f"Error processing query {i}: {e}")
                import traceback

                traceback.print_exc()
                continue

        print(f"\nFinal variant distribution:")
        total = sum(variant_counts.values())
        for v in sorted(variant_counts.keys()):
            pct = (variant_counts[v] / total) * 100 if total > 0 else 0
            print(f"  {v}: {variant_counts[v]} ({pct:.1f}%)")

        print(f"Collected {len(self.logs)} total interaction logs")
        csv_filename = f"logs_{self.ab_test_phase}.csv"
        self.save_logs_to_csv(csv_filename)

    def run_experiment(self, initial_queries=100, experiment_queries=200):

        def init_variant_metrics():
            return defaultdict(
                lambda: {
                    "clicks": [],
                    "dwell_times": [],
                    "likes": [],
                    "shares": [],
                    "bookmarks": [],
                    "weighted_clicks": [],
                    "users": set(),
                    "queries": 0,
                    "total_impressions": 0,
                    "actions_per_query": [],
                }
            )

        print("\n=== News Ranking A/B Testing Experiment ===")

        # Phase 1: Collect initial baseline data
        print("\n--- Phase 1: Initial Data Collection (Baseline Only) ---")
        self.ab_test_phase = "baseline"
        self.collect_data_with_ab_testing(num_queries=initial_queries)

        # Phase 2: Train all models
        print("\n--- Phase 2: Training Models ---")
        self.train_collaborative_filtering()
        self.train_ltr_model(iterations=100, learning_rate=0.01, propensity_clip=0.1)
        self.train_xgboost_model(propensity_clip=0.1)

        # Phase 3: Run A/B test
        print("\n--- Phase 3: A/B Testing ---")
        self.ab_test_phase = "experiment"

        # Reset variant metrics for clean experiment
        self.variant_metrics = init_variant_metrics()

        self.collect_data_with_ab_testing(num_queries=experiment_queries)

        # Phase 4: Analyze results
        print("\n--- Phase 4: Results Analysis ---")
        results = self.analyze_results_detailed(self.variant_metrics)

        return results

    def analyze_results_detailed(self, variant_metrics, epsilon=1e-6):

        print("\n" + "=" * 80)
        print("COMPREHENSIVE A/B TEST RESULTS")
        print("=" * 80)

        baseline = variant_metrics["baseline"]

        metrics_to_analyze = [
            ("CTR", "clicks", "total_impressions", True),  # rate
            ("Avg Dwell Time", "dwell_times", None, False),  # average
            ("Likes/Query", "likes", "queries", True),
            ("Shares/Query", "shares", "queries", True),
            ("Bookmarks/Query", "bookmarks", "queries", True),
            ("Weighted Clicks", "weighted_clicks", None, False),
        ]

        results = {}

        all_variants = ["baseline", "ltr", "xgboost", "collaborative"]
        for variant_name in all_variants:
            if variant_name not in variant_metrics:
                continue

            variant = variant_metrics[variant_name]
            results[variant_name] = {}

            if variant_name != "baseline":
                print(f"\n{'='*80}")
                print(f"{variant_name.upper()} vs BASELINE")
                print(f"{'='*80}")

            for metric_name, data_key, denominator_key, is_rate in metrics_to_analyze:

                # Smoothed metric values
                if is_rate and denominator_key:
                    baseline_val = (sum(baseline.get(data_key, [0])) + epsilon) / (
                        baseline.get(denominator_key, 1) + epsilon
                    )
                    variant_val = (sum(variant.get(data_key, [0])) + epsilon) / (
                        variant.get(denominator_key, 1) + epsilon
                    )
                else:
                    baseline_val = np.mean(baseline.get(data_key, [0])) + epsilon
                    variant_val = np.mean(variant.get(data_key, [0])) + epsilon

                # Improvement over baseline
                improvement = (
                    (variant_val - baseline_val) / (baseline_val + epsilon) * 100
                    if variant_name != "baseline"
                    else 0.0
                )

                # Statistical test
                if (
                    variant_name != "baseline"
                    and baseline.get(data_key)
                    and variant.get(data_key)
                ):
                    t_stat, p_value = stats.ttest_ind(
                        variant.get(data_key, [0]),
                        baseline.get(data_key, [0]),
                        equal_var=False,
                    )
                else:
                    t_stat, p_value = 0.0, 1.0

                results[variant_name][metric_name] = {
                    "baseline": baseline_val,
                    "variant": variant_val,
                    "improvement": improvement,
                    "p_value": p_value,
                    "significant_95": p_value < 0.05,
                    "t_statistic": t_stat,
                }

                if variant_name != "baseline":
                    print(f"\n{metric_name}:")
                    print(f"  Baseline: {baseline_val:.4f}")
                    print(f"  Variant:  {variant_val:.4f}")
                    print(f"  Change:   {improvement:+.2f}%")
                    print(f"  p-value:  {p_value:.6f}")
                    print(
                        f"  Significant (95%): {' Yes' if results[variant_name][metric_name]['significant_95'] else ' No'}"
                    )

        # --- Experiment Summary ---
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}")

        queries_processed = sum(v["queries"] for v in variant_metrics.values())
        self.monitor_ab_test_progress(queries_processed)

        # Use Weighted Clicks as main comparison
        baseline_mean = results["baseline"]["Weighted Clicks"]["baseline"]
        non_baseline_variants = [v for v in results if v != "baseline"]
        if non_baseline_variants:
            best_variant = max(
                [
                    (v, results[v]["Weighted Clicks"]["variant"])
                    for v in non_baseline_variants
                ],
                key=lambda x: x[1],
                default=("baseline", baseline_mean),
            )

            print(f"Best performing variant: {best_variant[0].upper()}")
            print(f"Best mean weighted clicks: {best_variant[1]:.4f}")

            significant_winners = [
                v
                for v in non_baseline_variants
                if results[v]["Weighted Clicks"]["significant_95"]
                and results[v]["Weighted Clicks"]["improvement"] > 0
            ]

            if significant_winners:
                print(
                    f"\n Significant improvements found: {', '.join(significant_winners)}"
                )
                print("Recommendation: Deploy best performing variant")
            else:
                print("\n No significant improvements found")
                print("Recommendation: Continue with baseline or collect more data")
        else:
            print("No variant data to compare")

        return results

    def save_model(self, filepath="ranking_model.pkl"):
        model_data = {
            "ltr_weights": self.ltr_weights,
            "xgb_model": self.xgb_model,
            "svd_model": self.svd_model,
            "user_id_map": self.user_id_map,
            "article_id_map": self.article_id_map,
            "logs": self.logs,
            "user_histories": dict(self.user_histories),
            "variant_metrics": {
                k: {
                    "weighted_clicks": v["weighted_clicks"],
                    "users": list(v["users"]),
                    "queries": v["queries"],
                }
                for k, v in self.variant_metrics.items()
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Models saved to {filepath}")

    def load_model(self, filepath="ranking_model.pkl"):

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)
        self.ltr_weights = model_data["ltr_weights"]
        self.xgb_model = model_data.get("xgb_model")
        self.svd_model = model_data.get("svd_model")
        self.user_id_map = model_data.get("user_id_map", {})
        self.article_id_map = model_data.get("article_id_map", {})
        self.logs = model_data["logs"]
        self.user_histories = defaultdict(list, model_data["user_histories"])
        print(f"Models loaded from {filepath}")


if __name__ == "__main__":

    system = NewsRankingSystem(api_url="http://localhost:3000")

    system.load_articles("./data/articles.jsonl")
    system.index_articles_to_es()

    print("\nStarting News Ranking A/B Test Experiment")
    results = system.run_experiment(
        initial_queries=500,
        experiment_queries=200,
    )

    system.save_model("news_ranking_models.pkl")

    print("\nExperiment Complete!")
    print(f"Total logs collected: {len(system.logs)}")
    print(f"Unique users: {len(system.user_histories)}")
    print("\nModels trained:")
    print(f"  - LTR: {'✓' if system.ltr_weights is not None else '✗'}")
    print(f"  - XGBoost: {'✓' if system.xgb_model is not None else '✗'}")
    print(
        f"  - Collaborative Filtering: {'✓' if system.svd_model is not None else '✗'}"
    )

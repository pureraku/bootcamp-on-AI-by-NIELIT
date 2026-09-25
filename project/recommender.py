import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = Path("data/anime.csv")

class AnimeRecommender:
    def __init__(self, data_path=DATA_PATH):
        self.df = pd.read_csv(data_path)
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        self.df.columns = self.df.columns.str.strip().str.lower()

        required = ["name", "genre", "type", "synopsis", "rating", "members"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for col in ["name", "genre", "type", "synopsis"]:
            self.df[col] = self.df[col].fillna("").astype(str)

        for col in ["rating", "members"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

        # Give more influence to genre than general synopsis text.
        self.df["features"] = (
            self.df["genre"].str.replace(",", " ", regex=False) + " " +
            self.df["genre"].str.replace(",", " ", regex=False) + " " +
            self.df["type"] + " " +
            self.df["synopsis"]
        )

        self.df["name_key"] = self.df["name"].str.lower().str.strip()

    def _build_model(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["features"])
        self.content_similarity = cosine_similarity(self.tfidf_matrix)

        # Popularity/rating are normalized so they can be blended with content similarity.
        scaler = MinMaxScaler()
        self.df[["rating_norm", "members_norm"]] = scaler.fit_transform(
            self.df[["rating", "members"]]
        )

    def search(self, query, limit=10):
        query = query.lower().strip()
        results = self.df[
            self.df["name"].str.lower().str.contains(query, na=False)
        ][["name", "genre", "type", "rating"]].head(limit)
        return results

    def recommend(self, anime_name, n=10, content_weight=0.85):
        matches = self.df.index[self.df["name_key"] == anime_name.lower().strip()].tolist()

        if not matches:
            # Fallback to partial search
            partial = self.search(anime_name, limit=5)
            if partial.empty:
                raise ValueError(f"Anime '{anime_name}' was not found.")
            closest_name = partial.iloc[0]["name"]
            matches = self.df.index[self.df["name_key"] == closest_name.lower()].tolist()

        idx = matches[0]

        scores = self.content_similarity[idx].copy()

        # Blend content similarity with rating and popularity.
        blended = (
            content_weight * scores
            + 0.10 * self.df["rating_norm"].to_numpy()
            + 0.05 * self.df["members_norm"].to_numpy()
        )

        blended[idx] = -1  # Don't recommend the input anime itself.

        top_indices = np.argsort(blended)[::-1][:n]

        results = self.df.iloc[top_indices][
            ["name", "genre", "type", "rating", "members"]
        ].copy()

        results["similarity"] = scores[top_indices]
        results["score"] = blended[top_indices]

        return results.reset_index(drop=True)

    def recommend_by_genre(self, genre, n=10):
        mask = self.df["genre"].str.contains(
            genre, case=False, na=False
        )

        return self.df.loc[mask, ["name", "genre", "type", "rating", "members"]] \
            .sort_values(["rating", "members"], ascending=False) \
            .head(n) \
            .reset_index(drop=True)


def print_recommendations(results):
    print("\n" + "=" * 75)
    print("RECOMMENDATIONS")
    print("=" * 75)

    for i, row in results.iterrows():
        print(
            f"{i + 1:2}. {row['name']}\n"
            f"    Genre: {row['genre']}\n"
            f"    Type: {row['type']} | Rating: {row['rating']}\n"
            f"    Content similarity: {row.get('similarity', 0):.3f}\n"
        )


def main():
    recommender = AnimeRecommender()

    print("\n🎌 ANIME RECOMMENDATION SYSTEM 🎌")
    print(f"Loaded {len(recommender.df)} anime.")
    print("\nCommands:")
    print("  recommend <anime name>")
    print("  search <text>")
    print("  genre <genre>")
    print("  quit")

    while True:
        command = input("\n> ").strip()

        if command.lower() == "quit":
            print("Goodbye!")
            break

        if command.lower().startswith("recommend "):
            name = command[10:].strip()
            try:
                results = recommender.recommend(name, n=10)
                print_recommendations(results)
            except ValueError as e:
                print(e)

        elif command.lower().startswith("search "):
            query = command[7:].strip()
            results = recommender.search(query)
            print("\nSEARCH RESULTS")
            print(results.to_string(index=False))

        elif command.lower().startswith("genre "):
            genre = command[6:].strip()
            results = recommender.recommend_by_genre(genre)
            print("\nGENRE RESULTS")
            print(results.to_string(index=False))

        else:
            print("Unknown command. Try: recommend Naruto")


if __name__ == "__main__":
    main()

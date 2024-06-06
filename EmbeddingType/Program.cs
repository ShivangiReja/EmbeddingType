using System.Numerics;
using System.Text.Json;
using System.Text;
using OpenAI.Embeddings;
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Models;
using System.ClientModel;
using NUnit.Framework;

namespace EmbeddingType
{
    class Program
    {
        public static void Main(string[] args)
        {
            VectorSearch();
            OpenAIGetEmbedding();
        }

        public static void VectorSearch()
        {
            Uri endpoint = new(Environment.GetEnvironmentVariable("SEARCH_ENDPOINT"));
            AzureKeyCredential credential = new(Environment.GetEnvironmentVariable("SEARCH_API_KEY"));
            string indexName = "mysearchindex";

            SearchClient searchClient = new SearchClient(endpoint, indexName, credential);
            ReadOnlyMemory<float> vectorizedResult = VectorSearchEmbeddings.SearchVectorizeDescription; // "Top hotels in town"

            Response<SearchResults<Hotel>> response = searchClient.Search<Hotel>(
                    new SearchOptions
                    {
                        VectorSearch = new()
                        {
                            Queries = { new VectorizedQuery(vectorizedResult) { KNearestNeighborsCount = 3, Fields = { "DescriptionVector" } } }
                        }
                    });

            ReadOnlyMemory<float> expactedVectors = response.Value.GetResults().First().Document.DescriptionVector;

            // Get the first raw document of the search result and deserialize the vector property.
            using JsonDocument jsonDoc = JsonDocument.Parse(response.GetRawResponse().Content.ToStream());
            JsonElement doc = jsonDoc.RootElement.EnumerateObject().First().Value.EnumerateArray().First();

            if (doc.TryGetProperty("DescriptionVector", out JsonElement descriptionVector))
            {
                EmbeddingVector vector = EmbeddingVector.FromJson(Encoding.UTF8.GetBytes(descriptionVector.GetRawText()));
                EmbeddingVector<float> floats = vector.To<float>();

                // Check if the decoded embedding matches the expected values
                bool result = floats.Scalars.Span.SequenceEqual(expactedVectors.Span);
                Assert.IsTrue(result);
            }
        }

        public static void OpenAIGetEmbedding()
        {
            EmbeddingClient client = new("text-embedding-ada-002", Environment.GetEnvironmentVariable("OPENAI-API-KEY"));

            string description = "Hello world";

            ClientResult<Embedding> response = client.GenerateEmbedding(description);

            ReadOnlyMemory<float> expactedVectors = response.Value.Vector;

            using JsonDocument jsonDoc = JsonDocument.Parse(response.GetRawResponse().Content.ToStream());
            JsonElement root = jsonDoc.RootElement;

            // Navigate to the "embedding" value
            if (root.TryGetProperty("data", out JsonElement dataElement) && dataElement.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement element in dataElement.EnumerateArray())
                {
                    if (element.TryGetProperty("embedding", out JsonElement embeddingElement))
                    {
                        // Convert the Base64-encoded embedding to an EmbeddingVector
                        EmbeddingVector vector = EmbeddingVector.FromBase64(Encoding.UTF8.GetBytes(embeddingElement.GetString()));

                        // Convert the embedding to EmbeddingVector<float>
                        EmbeddingVector<float> floatVector = vector.To<float>();

                        // Check if the decoded embedding matches the expected values
                        bool result = floatVector.Scalars.Span.SequenceEqual(expactedVectors.Span);
                        Assert.IsTrue(result);
                    }
                }
            }
        }
    }
}


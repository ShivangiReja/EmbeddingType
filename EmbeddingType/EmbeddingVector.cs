// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Text.Json;

namespace System.Numerics
{
    public abstract class EmbeddingVector
    {
        public virtual EmbeddingVector<T> To<T>() => throw new NotSupportedException();

        public static EmbeddingVector FromJson(ReadOnlyMemory<byte> utf8Json) => new JsonArrayVector(utf8Json);

        public static EmbeddingVector FromBase64(string base64) => new Base64Vector(base64);

        public static EmbeddingVector<T> FromScalars<T>(ReadOnlyMemory<T> scalars) => new EmbeddingVector<T>(scalars);

        public abstract void Write(Utf8JsonWriter writer, string format);
    }

    public sealed class EmbeddingVector<T> : EmbeddingVector
    {
        private readonly ReadOnlyMemory<T> _scalars;

        internal EmbeddingVector(ReadOnlyMemory<T> scalars) => _scalars = scalars;

        public ReadOnlyMemory<T> Scalars => _scalars;

        public override void Write(Utf8JsonWriter writer, string format)
        {
            if (!format.Equals("J", StringComparison.Ordinal))
            {
                throw new NotSupportedException();
            }

            JsonSerializer.Serialize(writer, _scalars.ToArray());
        }
    }

    internal sealed class JsonArrayVector : EmbeddingVector
    {
        private readonly ReadOnlyMemory<byte> _utf8Json;

        public JsonArrayVector(ReadOnlyMemory<byte> utf8Json) => _utf8Json = utf8Json;

        public override EmbeddingVector<T> To<T>()
        {
            try
            {
                var array = JsonSerializer.Deserialize<T[]>(_utf8Json.Span);
                return new EmbeddingVector<T>(array);
            }
            catch (Exception e)
            {
                throw new NotSupportedException(e.Message, e);
            }
        }

        public override void Write(Utf8JsonWriter writer, string format)
        {
            if (!format.Equals("J", StringComparison.Ordinal))
            {
                throw new NotSupportedException();
            }

            writer.WriteRawValue(_utf8Json.Span);
        }
    }

    internal sealed class Base64Vector : EmbeddingVector
    {
        private readonly string _base64;

        public Base64Vector(string base64) => _base64 = base64;

        public override EmbeddingVector<T> To<T>()
        {
            try
            {
                var decodedBytes = Convert.FromBase64String(_base64);
                var array = JsonSerializer.Deserialize<T[]>(decodedBytes);
                return new EmbeddingVector<T>(array);
            }
            catch (Exception e)
            {
                throw new NotSupportedException(e.Message, e);
            }
        }

        public override void Write(Utf8JsonWriter writer, string format)
        {
            if (!format.Equals("J", StringComparison.Ordinal))
            {
                throw new NotSupportedException();
            }

            writer.WriteStringValue(_base64);
        }
    }
}

using System.Buffers.Binary;
using System.Text;
using System.Text.Json;

namespace System.Numerics
{
    public abstract class EmbeddingVector
    {
        public virtual EmbeddingVector<T> To<T>() => throw new NotSupportedException();

        public static EmbeddingVector FromJson(ReadOnlyMemory<byte> utf8Json) => new JsonArrayVector(utf8Json);

        public static EmbeddingVector FromBase64(ReadOnlyMemory<byte> utf8Base64) => new Base64Vector(utf8Base64);

        public static EmbeddingVector<T> FromScalars<T>(ReadOnlyMemory<T> scalars) => new EmbeddingVector<T>(scalars);

        public abstract void Write(Stream stream, string format);
    }

    public sealed class EmbeddingVector<T> : EmbeddingVector
    {
        private readonly ReadOnlyMemory<T> _scalars;

        internal EmbeddingVector(ReadOnlyMemory<T> scalars) => _scalars = scalars;

        public ReadOnlyMemory<T> Scalars => _scalars;

        public override void Write(Stream stream, string format)
        {
            if (!format.Equals("J", StringComparison.Ordinal))
            {
                throw new NotSupportedException();
            }

            using var writer = new Utf8JsonWriter(stream);
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

        public override void Write(Stream stream, string format)
        {
            if (!format.Equals("J", StringComparison.Ordinal))
            {
                throw new NotSupportedException();
            }

            using var writer = new Utf8JsonWriter(stream);
            writer.WriteRawValue(_utf8Json.Span);
        }
    }

    internal sealed class Base64Vector : EmbeddingVector
    {
        private readonly ReadOnlyMemory<byte> _utf8Base64;

        public Base64Vector(ReadOnlyMemory<byte> utf8Base64) => _utf8Base64 = utf8Base64;

        public override EmbeddingVector<T> To<T>()
        {
            try
            {
                var decodedBytes = Convert.FromBase64String(Encoding.UTF8.GetString(_utf8Base64.Span));
                var span = new ReadOnlySpan<byte>(decodedBytes);

                T[] array;

                if (typeof(T) == typeof(float))
                {
                    array = new T[span.Length / sizeof(float)];
                    for (int i = 0; i < array.Length; i++)
                    {
                        float value = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(i * sizeof(float), sizeof(float)));
                        array[i] = (T)(object)value;
                    }
                }
#if NET5_0_OR_GREATER
                else if (typeof(T) == typeof(Half))
                {
                    array = new T[span.Length / sizeof(short)];
                    for (int i = 0; i < array.Length; i++)
                    {
                        Half value = (Half)BinaryPrimitives.ReadSingleLittleEndian(span.Slice(i * sizeof(short), sizeof(short)));
                        array[i] = (T)(object)value;
                    }
                }
#endif
                else
                {
                    throw new NotSupportedException($"Type {typeof(T)} is not supported.");
                }

                return new EmbeddingVector<T>(array);
            }
            catch (Exception e)
            {
                throw new NotSupportedException(e.Message, e);
            }
        }

        public override void Write(Stream stream, string format)
        {
            if (!format.Equals("J", StringComparison.Ordinal))
            {
                throw new NotSupportedException();
            }

            using var writer = new Utf8JsonWriter(stream);
            writer.WriteStringValue(_utf8Base64.Span);
        }
    }
}

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re


def split_into_sentences(text):
    """
    텍스트를 문장 단위로 분할
    
    Args:
        text (str): 마크다운 텍스트
    
    Returns:
        list: 문장 리스트
    """
    # 기본 문장 분리 (., !, ? 기준)
    # 마크다운 헤더나 특수 구조는 보존
    sentences = []
    
    # 줄 단위로 먼저 분리
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 마크다운 헤더는 그대로 유지
        if line.startswith('#'):
            sentences.append(line)
        # 코드 블록, 리스트는 그대로 유지
        elif line.startswith('```') or line.startswith('-') or line.startswith('*'):
            sentences.append(line)
        # 일반 텍스트는 문장 단위로 분리
        else:
            # 문장 부호로 분리
            parts = re.split(r'([.!?]\s+)', line)
            current = ""
            for i in range(0, len(parts)-1, 2):
                if i+1 < len(parts):
                    sentence = parts[i] + parts[i+1]
                    sentence = sentence.strip()
                    if sentence:
                        sentences.append(sentence)
            # 마지막 부분 처리
            if len(parts) % 2 == 1 and parts[-1].strip():
                sentences.append(parts[-1].strip())
    
    return sentences


def semantic_chunking(sentences, embeddings_model, threshold=0.5):
    """
    임베딩 유사도 기반 시맨틱 청킹
    
    Args:
        sentences (list): 문장 리스트
        embeddings_model: SentenceTransformer 모델
        threshold (float): 유사도 임계값 (0~1)
    
    Returns:
        list: 청크 리스트 (각 청크는 문장들의 리스트)
    """
    if not sentences:
        return []
    
    # 모든 문장 임베딩 계산
    print("문장 임베딩 계산 중...")
    embeddings = embeddings_model.encode(sentences, convert_to_tensor=True)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0].unsqueeze(0)  # 첫 문장 임베딩
    
    print(f"\n시맨틱 청킹 시작 (threshold: {threshold})")
    
    for i in range(1, len(sentences)):
        # 현재 청크와 다음 문장의 유사도 계산
        next_embedding = embeddings[i].unsqueeze(0)
        similarity = util.cos_sim(current_embedding, next_embedding).item()
        
        print(f"[{i}/{len(sentences)}] 유사도: {similarity:.3f}", end="")
        
        if similarity >= threshold:
            # 유사도가 높으면 현재 청크에 추가
            current_chunk.append(sentences[i])
            # 현재 청크의 평균 임베딩 업데이트
            current_embedding = embeddings[i-len(current_chunk)+1:i+1].mean(dim=0).unsqueeze(0)
            print(f" → 청크에 추가 (현재 청크 크기: {len(current_chunk)})")
        else:
            # 유사도가 낮으면 새 청크 시작
            chunks.append(current_chunk)
            current_chunk = [sentences[i]]
            current_embedding = next_embedding
            print(f" → 새 청크 시작 (총 청크 수: {len(chunks)})")
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def pdf_to_markdown_chunks(pdf_path, threshold=0.5):
    """
    PDF를 마크다운으로 변환 후 시맨틱 청킹
    
    Args:
        pdf_path (str): PDF 파일 경로
        threshold (float): 유사도 임계값
    
    Returns:
        list: 청크 리스트 (각 청크는 문장들을 결합한 텍스트)
    """
    # PDF를 마크다운으로 변환
    print("PDF를 마크다운으로 변환 중...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(pdf_path)
    full_text = result.document.export_to_markdown(image_mode="referenced")
    
    # 문장 단위로 분할
    print("\n문장 단위로 분할 중...")
    sentences = split_into_sentences(full_text)
    print(f"총 {len(sentences)}개 문장 추출")
    
    # 임베딩 모델 로드
    print("\n임베딩 모델 로드 중...")
    embeddings_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    
    # 시맨틱 청킹
    chunk_sentences = semantic_chunking(sentences, embeddings_model, threshold)
    
    # 청크를 텍스트로 결합
    chunks = []
    for i, chunk in enumerate(chunk_sentences):
        chunk_text = '\n'.join(chunk)
        chunks.append(chunk_text)
        print(f"\n{'='*80}")
        print(f"[청크 {i+1}/{len(chunk_sentences)}] (문장 수: {len(chunk)}, 길이: {len(chunk_text)}자)")
        print(f"{'-'*80}")
        print(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)
    
    return chunks


def save_chunks_simple(chunks, output_path="chunks_output.txt"):
    """청크를 텍스트 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"총 {len(chunks)}개 청크\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            f.write(f"{'='*80}\n")
            f.write(f"[청크 {i}/{len(chunks)}]\n")
            f.write(f"{'-'*80}\n")
            f.write(chunk)
            f.write(f"\n{'='*80}\n\n")
    
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    pdf_path = "./test_data/safe_sql.pdf"
    
    # 시맨틱 청킹 (threshold 조절 가능)
    chunks = pdf_to_markdown_chunks(pdf_path, threshold=0.5)
    
    # 결과 저장
    save_chunks_simple(chunks, "safe_sql_semantic_chunks.txt")
    
    # 통계
    print(f"\n 청킹 통계")
    print(f"- 총 청크 수: {len(chunks)}")
    print(f"- 평균 길이: {sum(len(c) for c in chunks) / len(chunks):.0f}자")
    print(f"- 최소 길이: {min(len(c) for c in chunks)}자")
    print(f"- 최대 길이: {max(len(c) for c in chunks)}자")

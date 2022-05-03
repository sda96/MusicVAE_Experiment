# A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music

## Problem

The Variational Autoencoder (VAE) has proven to be an effective model for producing semantically meaningful latent representations for natural data

VAE는 데이터에서 의미가 담긴 잠재 표현을 생성하는데 효율적인 모델이라는 것이 증명되어져 왔습니다.

- **Why use VAE ?**

  Fundamentally, our model is an autoencoder, i.e., its goal is to accurately reconstruct its inputs. However, we additionally desire the ability to draw novel samples and perform latent-space interpolations and attribute vector arithmetic. For these properties, we adopt the framework of the Variational Autoencoder

  기본적으로, 연구진의 모델은 autoencoder인데 연구의 목적은 모델의 입력들을 재구축하는 것입니다. 하지만, 연구진은 추가적으로 novel sample을 만드는 능력과 잠재-공간 보간법과 산수가 되는 벡터로 변환되기를 원했습니다. 이러한 특성 때문에 연구진의 프레임워크는 VAE로 정하게 되었습니다.

  

However, it has thus far seen limited application to sequential data, and, as we demonstrate, **existing recurrent VAE models have difficulty modeling sequences with long-term structure**

하지만, 시퀀스한 데이터에서는 제한된 성능을 보여주고 있습니다, 그리고 연구진이 설명했다시피, **현존하는 재귀형 VAE 모델들은 긴-문장 구조를 가진 시퀀스 모델링에 어려움을 겪고 있습니다.**

## Action(Method, Contribution)

**we propose the use of a hierarchical decoder**, which first outputs embeddings for subsequences of the input and then uses these embeddings to generate each subsequence independently.

연구진은 처음에는 입력의 일부 시퀀스를 위한 임베딩으로 출력하고나서 이러한 임베딩을 각 일부 시퀀스를 독립적으로 생성하는데 사용되는 **계층적 디코더를 제안합니다.**

- **Why use hierarchical decoder?**

  the decoder in a recurrent VAE is typically a simple stacked RNN. The decoder RNN uses the latent vector z to set its initial state, and proceeds to generate the output sequence autoregressively. In preliminary experiments (discussed in Section 5), **we found that using a simple RNN as the decoder resulted in poor sampling and reconstruction for long sequences.** We believe this is caused by the vanishing influence of the latent state as the output sequence is generated.

  재귀성 VAE에 잇는 디코더는 일반적으로 간단한 스택형 RNN입니다. 디코더 RNN은 잠재 벡터 z를 초기 state로 사용하고, 자기회귀성을 지닌 출력 시퀀스를 생성하는데 처리됩니다. Section 5에서의 실험을 보면, **연구진은 간단한 RNN을 디코더로 사용하면 긴 시퀀스를 재구축하는데 poor sampling한 결과를 만드는 것을 알게되었습니다.** 연구진은 이 원인이 생성되어진 출력 시퀀스에 있는 잠재 state가 vanishing influence 때문이라고 생각합니다.

- **What is a hierarchical decoder?**

  ![image](https://user-images.githubusercontent.com/51338268/166414462-13aed968-28ab-4e99-9daa-962e2389c2c6.png)

  - **Conductor RNN**

    -  Assume that the input sequence (and target output sequence) $\bold{x}$ can be segmented into $U$ **nonoverlapping subsequences $y_u$** with endpoints $i_u$ so that where we define the special case of $i_{U+1} = T$. 

      ![image](https://user-images.githubusercontent.com/51338268/166436383-94ffe14d-4ea0-48e3-8dd3-3c89497e570f.png)

      입력 시퀀스(그리고 목적, 출력 시퀀스) $\bold{x}$를 **겹치지 않는 부분 시퀀스 $y_u$**로  끝점 $i_u$ 기준으로 분할이 가능합니다. 그리고 연구진은 특별한 경우는 $i_{U+1} = T$ 로 정의 하겠습니다.

      Then, the latent vector z is passed through a fully-connected layer followed by a tanh activation to get the initial state of a “conductor” RNN. 

      그리고 잠재벡터 z는 "conductor RNN"의 초기 state를 얻기 위해서 완전결합층을 지나 tanh 활성화 함수가 적용되어집니다.

      **The conductor RNN produces $U$ embedding vectors $\bold{c} = \{c_1, c_2, . . . , c_U \}$, one for each subsequence.** In our experiments, we use a two-layer unidirectional LSTM for the conductor with a hidden state size of 1024 and 512 output dimensions.

      **"conductor RNN"은 U 각 시퀀스마다 하나의 임베딩 벡터를 생성합니다.** 연구진의 실험에서는 은닉층의 크기가 1024이고 출력 차원이 512인 2개의 단방향 LSTM을 사용하였습니다.

  - **Decoder RNN**

    - Once the conductor has produced the sequence of embedding vectors $c$, each one is individually passed through a shared fully-connected layer followed by a tanh activation to produce initial states for a final bottom-layer decoder RNN. 

      일단 "conductor RNN"은 시퀀스마다 임베딩 벡터 $c$를 만들어냅니다, 각각의 시퀀스는 개별적으로 지나가며 공유되는 완전결합층을 지나 tanh를 지나 결과적으로는 "decoder RNN" 초기 state에 적용이 되어집니다.

      **The decoder RNN then autoregressively produces a sequence of distributions over output tokens for each subsequence $y_{u}$ via a softmax output layer.** 

      **"decoder RNN" 자기회귀적으로 softmax 출력층을 통하여 각각의 부분 시퀀스 $y_u$ 에 대한 출력 토큰들의 분포 시퀀스를 생성합니다.**

      At each step of the bottom-level decoder, the current conductor embedding $c_u$ is concatenated with the previous output token to be used as the input. 

      bottom-level decoder의 각각의 단계는 현재 conductor embedding인 $c_u$는 이전 출력 토큰은 입력토큰과 결합되어집니다.

      In our experiments, we used a 2-layer LSTM with 1024 units per layer for the decoder RNN.

      연구진이 한 실험에서는 1024 unit을 가진 2층 LSTM을 각 층마다 사용하였습니다.

  - **“posterior collapse” problem**

    - posterior collapse 문제는 디코더가 인코더의 condition을 무시하고 output을 생성하는 문제를 말합니다
    
      [참고사이트](https://stopspoon.tistory.com/63)
    
    - **we find that it is important to limit the scope of the decoder** to force it to use the latent code to model long-term structure.
    
      연구진은 긴-문장 구조를 모델링하기 위해서 잠재 코드를 사용하는데 **디코더의 범위를 제한하는 것이 중요하다는 것을 발견했습니다.**
    
      For a CNN decoder, this is as simple as reducing the receptive field, but no direct analogy exists for RNNs, which in principle have an unlimited temporal receptive field.
    
      CNN 디코더에서는 수용 영역을 줄이는 것으로 간단하지만,  RNN에서는 직접적으로 유사한 것은 없기에 해당 원리는 제한 없는 임시 수용 영역을 가져야 합니다.
    
      To get around this, **we reduce the effective scope of the bottom-level RNN in the decoder by only allowing it to propagate state within an output subsequence.**
    
      그러기 위해서, **연구진은 디코더에서 오직 출력 부분 시퀀스안의 state만 전파를 허락함으로써 bottom-level RNN 효율적인 범위를 감소시켰습니다.**

This structure encourages the model to utilize its latent code, thereby avoiding the “posterior collapse” problem, which remains an issue for recurrent VAEs.

이러한 구조는 모델이 자체적 잠재된 코드를 사용하도록 격려하였고, 이 방법으로 VAEs가 겪고 있던 문제인 "사후 충돌" 문제를 피하게 해줍니다.

## Result(Experiment, Conclusion)

We apply this architecture to modeling sequences of musical notes and find that it exhibits dramatically better sampling, interpolation, and reconstruction performance than a “flat” baseline model.

연구진은 해당 구조를 음악 노트 시퀀스 모델링에 적용하였고, 기존 "flat" 기본 모델보다 드라미틱컬하게 좋아진 sampling, interpolation, reconstruction performance를 보여주었습니다.
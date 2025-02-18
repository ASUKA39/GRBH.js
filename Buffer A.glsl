#define PI 3.141592653589
#define G0 6.673e-11
#define lightspeed 299792458.0
#define sigma 5.670373e-8
#define ly 9460730472580800.0
#define Msun 1.9891e30
#define FOV 0.5
float RandomStep(vec2 xy, float seed)//用于光线起点抖动的随机
{
    return fract(sin(dot(xy.xy+fract(11.4514*sin(seed)), vec2(12.9898, 78.233)))* 43758.5453);
}

float cubicInterpolation(float t)//从0到1插值,用于perlin noise
{
//return t * t * t * (t * (t * 6. - 15.) + 10.);
return 3.*t*t-2.*t*t*t;
}

//float PerlinNoise(vec3 xyz)//三维perlin noise,用于吸积盘云的随机,这个快一些但是云场有网状纹，原因不明
//{
//    ivec3 p000=ivec3(int(floor(xyz.x)),int(floor(xyz.y)),int(floor(xyz.z)));
//    int effp00=129*(p000.x    )+782*(p000.y    )+213*(p000.z    );
//    float v000=2.0*fract(sin(float(effp00                                            ))* 43758.5453)-1.0;
//    float v100=2.0*fract(sin(float(effp00+129                                        ))* 43758.5453)-1.0;
//    float v010=2.0*fract(sin(float(effp00    +782                                    ))* 43758.5453)-1.0;
//    float v110=2.0*fract(sin(float(effp00+129+782                                    ))* 43758.5453)-1.0;
//    float v001=2.0*fract(sin(float(effp00         +213                               ))* 43758.5453)-1.0;
//    float v101=2.0*fract(sin(float(effp00+129     +213                               ))* 43758.5453)-1.0;
//    float v011=2.0*fract(sin(float(effp00    +782 +213                               ))* 43758.5453)-1.0;
//    float v111=2.0*fract(sin(float(effp00+129+782 +213                               ))* 43758.5453)-1.0;
//    vec3 pf=vec3(fract(xyz.x),fract(xyz.y),fract(xyz.z));
//    float v00=v001*cubicInterpolation(pf.z)+v000*cubicInterpolation(1.0-pf.z);
//    float v10=v101*cubicInterpolation(pf.z)+v100*cubicInterpolation(1.0-pf.z);
//    float v01=v011*cubicInterpolation(pf.z)+v010*cubicInterpolation(1.0-pf.z);
//    float v11=v111*cubicInterpolation(pf.z)+v110*cubicInterpolation(1.0-pf.z);
//    float v0=v01*cubicInterpolation(pf.y)+v00*cubicInterpolation(1.0-pf.y);
//    float v1=v11*cubicInterpolation(pf.y)+v10*cubicInterpolation(1.0-pf.y);
//    return v1*cubicInterpolation(pf.x)+v0*cubicInterpolation(1.0-pf.x);
//}


float PerlinNoise(vec3 xyz)//三维perlin noise,用于吸积盘云的随机
{
    vec3 p000=vec3(floor(xyz.x),floor(xyz.y),floor(xyz.z));
    float v000=2.0*fract(sin(dot(vec3(p000.x    ,p000.y    ,p000.z    ), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    float v100=2.0*fract(sin(dot(vec3(p000.x+1.0,p000.y    ,p000.z    ), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    float v010=2.0*fract(sin(dot(vec3(p000.x    ,p000.y+1.0,p000.z    ), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    float v110=2.0*fract(sin(dot(vec3(p000.x+1.0,p000.y+1.0,p000.z    ), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    float v001=2.0*fract(sin(dot(vec3(p000.x    ,p000.y    ,p000.z+1.0), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    float v101=2.0*fract(sin(dot(vec3(p000.x+1.0,p000.y    ,p000.z+1.0), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    float v011=2.0*fract(sin(dot(vec3(p000.x    ,p000.y+1.0,p000.z+1.0), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    float v111=2.0*fract(sin(dot(vec3(p000.x+1.0,p000.y+1.0,p000.z+1.0), vec3(12.9898, 78.233,213.765)))* 43758.5453)-1.0;
    vec3 pf=vec3(fract(xyz.x),fract(xyz.y),fract(xyz.z));
    float v00=v001*cubicInterpolation(pf.z)+v000*cubicInterpolation(1.0-pf.z);
    float v10=v101*cubicInterpolation(pf.z)+v100*cubicInterpolation(1.0-pf.z);
    float v01=v011*cubicInterpolation(pf.z)+v010*cubicInterpolation(1.0-pf.z);
    float v11=v111*cubicInterpolation(pf.z)+v110*cubicInterpolation(1.0-pf.z);
    float v0=v01*cubicInterpolation(pf.y)+v00*cubicInterpolation(1.0-pf.y);
    float v1=v11*cubicInterpolation(pf.y)+v10*cubicInterpolation(1.0-pf.y);
    return v1*cubicInterpolation(pf.x)+v0*cubicInterpolation(1.0-pf.x);
}



float softHold(float x){//使不大于一
return 1.0-1.0/(max(x,0.0)+1.0);
}


float DiskRandom0(vec3 EPos,int granularityStart,int granularityEnd,float Contrast){//云场强噪声,输入为  位置（不一定是空间位置,可以经过变形）,细节起始级别,细节终止级别,场强强弱对比度
    float Result=10.;
    float Rate=1.;
    for(int i=granularityStart;i<granularityEnd;i++){
    Rate=pow(3.0,float(i)); Result*=(1.0+0.1*PerlinNoise(vec3(Rate*EPos.x,Rate*EPos.y,Rate*EPos.z))); 
    }
    return log(1.+pow(0.1*Result,Contrast));
}

float Vec2Theta(vec2 a,vec2 b)//两平面向量夹角,0到2pi
{
if(dot(a,b)>0.0){
  return asin(0.999999*(a.x*b.y-a.y*b.x)/length(a)/length(b));
 }else if(dot(a,b)<0.0 && (-a.x*b.y+a.y*b.x)<0.0){
 return PI-asin(0.999999*(a.x*b.y-a.y*b.x)/length(a)/length(b));
 }else if(dot(a,b)<0.0 && (-a.x*b.y+a.y*b.x)>0.0){
 return -PI-asin(0.999999*(a.x*b.y-a.y*b.x)/length(a)/length(b));
 }
}

vec3 RGB(float T) {
    if(T<400.01){return vec3(0.,0.,0.);}
    float _ = (T - 6500.0) / (6500.0 * T * 2.2);
    float R = exp(2.05539304e4 * _);
    float G = exp(2.63463675e4 * _);
    float B = exp(3.30145739e4 * _);
    float LmulRate = 1.0 / max(max(R, G), B);
    if(T<1000.){
    LmulRate*=(T-400.)/600.;
    }
    R *= LmulRate;
    G *= LmulRate;
    B *= LmulRate;
    return vec3(R, G, B);
}

float omega(float r,float Rs){//绕黑洞公转速度
return sqrt(lightspeed/ly*lightspeed*Rs/ly/((2.0*r-3.0*Rs)*r*r));
}

vec3 GetBH(vec4 a,vec3 BHPos,vec3 DiskDir)//BH系平移旋转
    {
    vec3 vecz =vec3( 0.0,0.0,1.0 );
    if(DiskDir==vecz){
    DiskDir+=0.0001*(vec3(1.0,0.,0.));
    }
     vec3 _X = normalize(cross(vecz, DiskDir));
     vec3 _Y = normalize(cross(DiskDir, _X));
     vec3 _Z = normalize(DiskDir);
    a=(transpose(mat4x4(
        1., 0., 0., -BHPos.x,
        0., 1., 0., -BHPos.y,
        0., 0., 1., -BHPos.z,
        0., 0., 0., 1.
    ))*a);
    a=transpose(mat4x4(
        _X.x,_X.y,_X.z,0.,
        _Y.x,_Y.y,_Y.z,0.,
        _Z.x,_Z.y,_Z.z,0.,
        0.   ,0.   ,0.   ,1.)
        )*a;
    return a.xyz;
}

vec3 GetBHRot(vec4 a,vec3 BHPos,vec3 DiskDir)//BH系旋转
    {
    vec3 vecz =vec3( 0.0,0.0,1.0 );
    if(DiskDir==vecz){
    DiskDir+=0.0001*(vec3(1.0,0.,0.));
    }
    vec3 _X = normalize(cross(vecz, DiskDir));
    vec3 _Y = normalize(cross(DiskDir, _X));
    vec3 _Z = normalize(DiskDir);

    a=transpose(mat4x4(
        _X.x,_X.y,_X.z,0.,
        _Y.x,_Y.y,_Y.z,0.,
        _Z.x,_Z.y,_Z.z,0.,
        0.   ,0.   ,0.   ,1.)
        )*a;
    return a.xyz;
}

vec4 GetCamera(vec4 a)//相机系平移旋转  本部分在实际使用时uniform输入
{
float _Theta=4.0*PI*iMouse.x/iResolution.x;
float _Phi=0.999*PI*iMouse.y/iResolution.y+0.0005;
float _R=0.000057;

if(iFrame<2){
    _Theta=4.0*PI*0.45;
    _Phi=0.999*PI*0.55+0.0005;
    
}
if(texelFetch(iChannel0, ivec2(83, 0), 0).x > 0.){
_R=0.000097;


}
if(texelFetch(iChannel0, ivec2(87, 0), 0).x > 0.){
_R=0.000017;


}
vec3 _Rotcen=vec3(0.0,0.0,0.0);

vec3 _Campos;

    vec3 reposcam=vec3(
    _R * sin(_Phi) * cos(_Theta),
    _R * sin(_Phi) * sin(_Theta),
    -_R * cos(_Phi));

    _Campos = _Rotcen + reposcam;
    vec3 vecz =vec3( 0.0,0.0,1.0 );

    vec3 _X = normalize(cross(vecz, reposcam));
    vec3 _Y = normalize(cross(reposcam, _X));
    vec3 _Z = normalize(reposcam);

    a=(transpose(mat4x4(
        1., 0., 0., -_Campos.x,
        0., 1., 0., -_Campos.y,
        0., 0., 1., -_Campos.z,
        0., 0., 0., 1.
    ))*a);
    
    a=transpose(mat4x4(
        _X.x,_X.y,_X.z,0.,
        _Y.x,_Y.y,_Y.z,0.,
        _Z.x,_Z.y,_Z.z,0.,
        0.   ,0.   ,0.   ,1.)
        )*a;
        
    return a;
}

vec4 GetCameraRot(vec4 a)//摄影机系旋转    本部分在实际使用时uniform输入
{
float _Theta=4.0*PI*iMouse.x/iResolution.x;
float _Phi=0.999*PI*iMouse.y/iResolution.y+0.0005;
float _R=0.000057;

if(iFrame<2){
    _Theta=4.0*PI*0.45;
    _Phi=0.999*PI*0.55+0.0005;
    
}
if(texelFetch(iChannel0, ivec2(83, 0), 0).x > 0.){
_R=0.000097;


}
if(texelFetch(iChannel0, ivec2(87, 0), 0).x > 0.){
_R=0.000017;


}
vec3 _Rotcen=vec3(0.0,0.0,0.0);

vec3 _Campos;

    vec3 reposcam=vec3(
    _R * sin(_Phi) * cos(_Theta),
    _R * sin(_Phi) * sin(_Theta),
    -_R * cos(_Phi));

    _Campos = _Rotcen + reposcam;
    vec3 vecz =vec3( 0.0,0.0,1.0 );

    vec3 _X = normalize(cross(vecz, reposcam));
    vec3 _Y = normalize(cross(reposcam, _X));
    vec3 _Z = normalize(reposcam);

    a=transpose(mat4x4(
        _X.x,_X.y,_X.z,0.,
        _Y.x,_Y.y,_Y.z,0.,
        _Z.x,_Z.y,_Z.z,0.,
        0.   ,0.   ,0.   ,1.)
        )*a;
    return a;
}

// Screen是中间0.5，横向0到1，纵向0.5-0.5*y/x到0.5+0.5*y/x,不是NDC    被这个注释的语句已经没了，但我不确定，所以留着这个注释

vec3 uvToDir(vec2 uv)                                                                                   //一堆坐标间变换
{
return normalize(vec3(FOV*(2.0*uv.x-1.0),FOV*(2.0*uv.y-1.0)*iResolution.y/iResolution.x,-1.0));
}

vec2 PosToNDC(vec4 pos)
{
return vec2(-pos.x/pos.z,-pos.y/pos.z*iResolution.x/iResolution.y);
}

vec2 DirToNDC(vec3 dir)
{
return vec2(-dir.x/dir.z,-dir.y/dir.z*iResolution.x/iResolution.y);
}

vec2 DirTouv(vec3 dir)
{
return vec2(0.5-0.5*dir.x/dir.z,0.5-0.5*dir.y/dir.z*iResolution.x/iResolution.y);
}

vec2 PosTouv(vec4 Pos)
{
return vec2(0.5-0.5*Pos.x/Pos.z,0.5-0.5*Pos.y/Pos.z*iResolution.x/iResolution.y);
}

float Shape( float x, float a, float b )//用于吸积盘截面轮廓
{
    float k = pow(a+b,a+b) / (pow(a,a)*pow(b,b));
    return k * pow( x, a ) * pow( 1.0-x, b );
}

vec4 diskcolor(vec4 fragColor,float timerate,float steplength,vec3 RayPos,vec3 lastRayPos,vec3 RayDir,vec3 lastRayDir,vec3 WorldZ,vec3 BHPos,vec3 DiskDir,float Rs,float RIn,float ROut,float diskA,float TPeak4,float shiftMax){//吸积盘
    vec3 CamOnDisk=GetBH(vec4(0.,0.,0.,1.0),BHPos,DiskDir);//黑洞系下相机位置
    vec3 References=GetBHRot(vec4(WorldZ,1.0),BHPos,DiskDir);//用于吸积盘角度零点确定
    vec3 PosOnDisk=GetBH(vec4(RayPos,1.0),BHPos,DiskDir);//光线黑洞系下位置
    vec3 DirOnDisk=GetBHRot(vec4(RayDir,1.0),BHPos,DiskDir);//光线黑洞系下方向
    
    // 此行以下在黑洞坐标系

    float PosR=length(PosOnDisk.xy);
    float PosZ=PosOnDisk.z;
    
    vec4 color=vec4(0.);
    if(abs(PosZ)<0.5*Rs && PosR<ROut && PosR>RIn){
         
        
        float EffR = 1.0-((PosR-RIn)/(ROut-RIn)*0.5);
        if((ROut-RIn)>9.0*Rs){//这个if用于大外径盘的厚度控制
            if(PosR<5.0*Rs+RIn){
            EffR = 1.0-((PosR-RIn)/(9.0*Rs)*0.5);
            }else{
            EffR = 1.0-(0.5/0.9*0.5+((PosR-RIn)/(ROut-RIn)-5.*Rs/(ROut-RIn))/(1.-5.*Rs/(ROut-RIn))*0.5);
            }
        }
        
        if((abs(PosZ)<0.5*Rs*Shape(EffR, 4.0, 0.9))||(PosZ<0.5*Rs*(1.-5.*pow(2.*(1.-EffR),2.)))){
            
            
            float omega0=omega(PosR,Rs);
            
            //本部分应挪出raymarching部分提前计算（待办
            float halfPiIn=PI/omega(3.0*Rs,Rs);
            float EffTime0=fract(iTime*timerate/(halfPiIn))*halfPiIn+    0.*halfPiIn;//所有成对出现01结尾的变量,都是用于两个吸积盘叠变防止过度缠绕
            float EffTime1=fract(iTime*timerate/(halfPiIn)+0.5)*halfPiIn+1.*halfPiIn;
            float Ntime0=trunc(iTime*timerate/(halfPiIn));
            float Ntime1=trunc(iTime*timerate/(halfPiIn)+0.5);
            float phase0=2.0*PI*fract(43758.5453*sin(Ntime0));//角度随机用于防止快倍率下过渡出现明显周期重复
            float phase1=2.0*PI*fract(43758.5453*sin(Ntime1));
            
            
            
            
            
            float rthe=Vec2Theta(PosOnDisk.xy,References.xy);
            float PosTheta=fract((rthe+omega0*EffTime0+phase0)/2./PI)*2.*PI;
            
            //计算盘温度
            float T=pow(diskA*Rs*Rs*Rs/(PosR*PosR*PosR)*max(1.0-sqrt(RIn/PosR),0.000001),0.25);
            //计算云相对速度
            vec3 v=ly/lightspeed*omega0*cross(vec3(0.,0.,1.),PosOnDisk);
            float vre=dot(-DirOnDisk,v);
            //计算多普勒因子
            float Dopler = sqrt((1.0+vre)/(1.0-vre));
            //总红移量,含多普勒因子和引力红移和
            float RedShift = Dopler*sqrt(max(1.0-Rs/PosR,0.000001))/sqrt(max(1.0-Rs/length(CamOnDisk),0.000001));
    
    
            
            float Rho;
            float Thick;
            float Vmix;
            vec4 color0=vec4(0.);
            vec4 color1=vec4(0.);
            float distcol;
            {
            Rho = Shape(EffR, 4.0, 0.9);
            if(abs(PosZ)<0.5*Rs*Rho){
                Thick = 0.5*Rs*Rho*(0.4+0.6*softHold(DiskRandom0(vec3(1.5*PosTheta,PosR/Rs,1.0),1,3,80.0)));//盘厚
                Vmix=max(0.,(1.0 - abs(PosZ) / Thick));
                Rho *= 0.7*Vmix*Rho;
                color0=vec4(DiskRandom0(vec3(1.*PosR/Rs,1.*PosZ/Rs,.5*PosTheta),3,6,80.0));//云本体
                color0.xyz*=Rho*1.4*(0.2+0.8*Vmix+(0.8-0.8*Vmix)*DiskRandom0(vec3(PosR/Rs,1.5*PosTheta,PosZ/Rs),1,3,80.0));
                color0.a*=(Rho);//*(1.0+Vmix);
            }
            if(abs(PosZ)<0.5*Rs*(1.-5.*pow(2.*(1.-EffR),2.))){
                distcol=max(1.-pow(PosZ/(0.5*Rs*max(1.-5.*pow(2.*(1.-EffR),2.),0.0001)),2.),0.)*DiskRandom0(vec3(1.5*fract((1.5*rthe+PI/halfPiIn*EffTime0+phase0)/2./PI)*2.*PI,PosR/Rs,PosZ/Rs),0,6,80.0);
                color0+=0.02*vec4(vec3(distcol),0.2*distcol)*sqrt(1.0001-DirOnDisk.z*DirOnDisk.z)*min(1.,Dopler*Dopler);
            }
            color0*=0.5-0.5*cos(2.*PI*fract(iTime*timerate/(halfPiIn)));//用于过渡
            }
            
            PosTheta=fract((rthe+omega0*EffTime1+phase1)/(2.*PI))*2.*PI;//更新相位
            
            {
            Rho = Shape(EffR, 4.0, 0.9);//同上
            if(abs(PosZ)<0.5*Rs*Rho){
                Thick = 0.5*Rs*Rho*(0.4+0.6*softHold(DiskRandom0(vec3(1.5*PosTheta,PosR/Rs,1.0),1,3,80.0)));
                Vmix=max(0.,(1.0 - abs(PosZ) / Thick));
                Rho *= 0.7*Vmix*Rho;
                color1=vec4(DiskRandom0(vec3(1.*PosR/Rs,1.*PosZ/Rs,.5*PosTheta),3,6,80.0));
                color1.xyz*=Rho*1.4*(0.2+0.8*Vmix+(0.8-0.8*Vmix)*DiskRandom0(vec3(PosR/Rs,1.5*PosTheta,PosZ/Rs),1,3,80.0));
                color1.a*=(Rho);//*(1.0+Vmix);
            }
            if(abs(PosZ)<0.5*Rs*(1.-5.*pow(2.*(1.-EffR),2.))){
                distcol=max(1.-pow(PosZ/(0.5*Rs*max(1.-5.*pow(2.*(1.-EffR),2.),0.0001)),2.),0.)*DiskRandom0(vec3(1.5*fract((1.5*rthe+PI/halfPiIn*EffTime1+phase1)/2./PI)*2.*PI,PosR/Rs,PosZ/Rs),0,6,80.0);
                color1+=0.02*vec4(vec3(distcol),0.2*distcol)*sqrt(1.0001-DirOnDisk.z*DirOnDisk.z)*min(1.,Dopler*Dopler);
            }
            color1*=0.5-0.5*cos(2.*PI*fract(iTime*timerate/(halfPiIn)+0.5));
            }
            
            color=color1+color0;
            color*=1.0+20.*exp(-10.*(PosR-RIn)/(ROut-RIn));//内侧增加密度
            // xyz亮度*密度  a密度
            
            float BrightWithoutRedshift=4.5*T*T*T*T/TPeak4;//原亮度
            if (T>1000.){T=max(1000.,T*RedShift*Dopler*Dopler);}
            //物理上严格的红移*多普勒高饱和度修正
            
            T=min(100000.0,T);
    
            color.xyz*=BrightWithoutRedshift*min(1.,1.8*(ROut-PosR)/(ROut-RIn))*RGB(T/exp((PosR-RIn)/(0.6*(ROut-RIn))));
            // 原始亮度*修正颜色(给温度乘一个指数下降,避免颜色过于单调)
            
            color.xyz*=min(shiftMax,RedShift)*min(shiftMax,Dopler);
            //原亮度修正*多普勒高对比度修正
            
            color.xyz*=pow((1.0-(1.0-min(1.,RedShift))*(PosR-RIn)/(ROut-RIn)),9.);//缝一个和红移与半径均有关的函数,使左右两侧的亮度下降不均,增加不对称性
            color.xyz*=min(1.,1.+0.5*((PosR-RIn)/RIn+RIn/(PosR-RIn))-max(1.,RedShift));//乘一个对勾函数,降低吸积盘中间部分的亮度,避免糊成一坨白色
            
            //步长积累
            color.xyz*=steplength /Rs ;
            color.a*=  steplength /Rs;
        }
    }

    return fragColor + color*(1.0-fragColor.a);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{   
    fragColor=vec4(0.,0.,0.,0.);
    vec2 uv = fragCoord/iResolution.xy;
   
    float timerate=30.;//本部分在实际使用时又uniform输入，此外所有iTime*timerate应替换为游戏内时间。
    
    float MBH=1.49e7;//单位是太阳质量                                                                                           本部分在实际使用时uniform输入
    float a0=0.0;//无量纲自旋系数                                                                                               本部分在实际使用时uniform输入
    float Rs=2.*MBH*G0/lightspeed/lightspeed*Msun;//单位是米                                                                   本部分在实际使用时uniform输入
    
    float z1=1.+pow(1.-a0*a0,0.333333333333333)*(pow(1.+a0*a0,0.333333333333333)+pow(1.-a0,0.333333333333333));//辅助变量      本部分在实际使用时uniform输入
    float RmsRatio=(3.+sqrt(3.*a0*a0+z1*z1)-sqrt((3.-z1)*(3.+z1+2.*sqrt(3.*a0*a0+z1*z1))))/2.;//赤道顺行最内稳定圆轨与Rs之比    本部分在实际使用时uniform输入
    float AccEff=sqrt(1.-1./RmsRatio);//吸积放能效率,以落到Rms为准                                                              本部分在实际使用时uniform输入
    
    float mu=1.;//吸积物的比荷的倒数,氕为1                                                                                      本部分在实际使用时uniform输入
    float dmdtEdd=6.327*mu/lightspeed/lightspeed*MBH*Msun/AccEff;//爱丁顿吸积率                                                本部分在实际使用时uniform输入
      
    float dmdt=(2e-6)*dmdtEdd;//吸积率                                                                                         本部分在实际使用时uniform输入
    
    float diskA=3.*G0*Msun/Rs/Rs/Rs*MBH*dmdt/(8.*PI*sigma);//吸积盘温度系数                                                     本部分在实际使用时uniform输入
    
    //计算峰值温度的四次方,用于自适应亮度。峰值温度出现在49RIn/36处
    float TPeak4=diskA*0.05665278;//                                                                                          本部分在实际使用时uniform输入
    
    Rs=Rs/ly;//单位是ly                                                                                                       本部分在实际使用时uniform输入
    float RIn=0.7*RmsRatio*Rs;//盘内缘,正常情况下等于最内稳定圆轨
    float ROut=12.*Rs;//盘外缘                                                                                                本部分在实际使用时uniform输入
    
    float shiftMax = 1.25;//设定一个蓝移的亮度增加上限,以免亮部过于亮                                                                
    
    vec3 WorldZ=GetCameraRot(vec4(0.,0.,1.,1.)).xyz;
    vec4 BHAPos=vec4(5.*Rs,0.0,0.0,1.0);//黑洞世界位置                                                                         本部分在实际使用时没有
    vec4 BHADiskDir=vec4(normalize(vec3(0.,.2,1.0)),1.0);//吸积盘世界法向                                                     本部分在实际使用时没有
    //以下在相机系
    vec3 BHRPos=GetCamera(BHAPos).xyz;//                                                                                     本部分在实际使用时uniform输入
    vec3 BHRDiskDir=GetCameraRot(BHADiskDir).xyz;//                                                                          本部分在实际使用时uniform输入
    vec3 RayDir=uvToDir(uv+0.5*vec2(RandomStep(uv, fract(iTime * 1.0+0.5)),RandomStep(uv, fract(iTime * 1.0)))/iResolution.xy);
    vec3 RayPos=vec3(0.0,0.0,0.0);
    vec3 lastRayPos;
    vec3 lastRayDir;
    vec3 PosToBH;
    vec3 NPosToBH;
    float steplength=0.;
    float lastR=length(PosToBH);
    float costheta;
    float dthe;    
    float dphirate;
    float dl;
    float Dis;
    bool flag=true;
    int count=0;
    while(flag==true){//测地raymarching
    
        PosToBH=RayPos-BHRPos;
        Dis=length(PosToBH);
        NPosToBH=PosToBH/Dis;
        
        if(Dis>(2.5*ROut) && Dis>lastR && count>50){//远离黑洞
        flag=false;
        uv=DirTouv(RayDir);
        //fragColor+=0.5*texelFetch(iChannel1, ivec2(vec2(fract(uv.x),fract(uv.y))*iChannelResolution[1].xy), 0 )*(1.0-fragColor.a);
        //fragColor+=vec4(.25)*(1.0-fragColor.a);
        
        }
        if(Dis<0.1*Rs){//命中奇点
        flag=false;
        //fragColor+=vec4(0.,1.,1.,1.)*(1.0-fragColor.a);
        }
        if(flag==true){
        fragColor=diskcolor(fragColor,timerate,steplength,RayPos,lastRayPos,RayDir,lastRayDir,WorldZ,BHRPos,BHRDiskDir,Rs,RIn,ROut,diskA,TPeak4,shiftMax);//吸积盘颜色
        }
        
        if(fragColor.a>0.99){//被完全遮挡
        flag=false;
        }
        lastRayPos=RayPos;
        lastRayDir=RayDir;
        lastR=Dis;
        costheta=length(cross(NPosToBH,RayDir));//前进方向与切向夹角
        dphirate=-1.0*costheta*costheta*costheta*(1.5*Rs/Dis);//单位长度光偏折角
        if(count==0){
        dl=RandomStep(uv, fract(iTime * 1.0));//光起步步长抖动
        }else{
        dl=1.0;
        }
        
        dl*=0.15+0.25*min(max(0.,0.5*(0.5*Dis/max(10.*Rs,ROut)-1.)),1.);
        
        if((Dis)>=2.0*ROut){//在吸积盘附近缩短步长。步长作为位置的函数必须连续,最好高阶可导,不然会造成光线上步前缘与下步后缘不重合,产生条纹
        dl*=Dis;
        }else if((Dis)>=1.0*ROut){
        dl*=(max(abs(dot(BHRDiskDir,PosToBH)),Rs)*(2.0*ROut-Dis)+Dis*(Dis-ROut))/ROut;
        }else if((Dis)>=RIn){
        dl*=max(abs(dot(BHRDiskDir,PosToBH)),Rs);
        }else if((Dis)>2.*Rs){
        dl*=(max(abs(dot(BHRDiskDir,PosToBH)),Rs)*(Dis-2.0*Rs)+Dis*(RIn-Dis))/(RIn-2.0*Rs);
        }else{
        dl*=Dis;
        } 
        

        RayPos+=RayDir*dl;
        dthe=dl/Dis*dphirate;
        RayDir=normalize(RayDir+(dthe+dthe*dthe*dthe/3.0)*cross(cross(RayDir,NPosToBH),RayDir)/costheta);//更新方向，里面的（dthe +dthe^3/3）是tan（dthe）
        steplength=length(RayPos-lastRayPos);
        
        count++;
        
        
    }
    //为了套bloom先逆处理一遍
    float colorRFactor=fragColor.r/fragColor.g;
    float colorBFactor=fragColor.b/fragColor.g;
    
    float bloomMax = 12.0;
    fragColor.r=min(-4.0*log(1.-pow(fragColor.r,2.2)),bloomMax*colorRFactor);
    fragColor.g=min(-4.0*log(1.-pow(fragColor.g,2.2)),bloomMax);
    fragColor.b=min(-4.0*log(1.-pow(fragColor.b,2.2)),bloomMax*colorBFactor);
    fragColor.a=min(-4.0*log(1.-pow(fragColor.a,2.2)),4.0);
       
    //TAA

    float blendWeight = 1.0-pow(0.5,(iTimeDelta)/max(min((0.131*36.0/(timerate)*(omega(3.*0.00000465,0.00000465))/(omega(3.*Rs,Rs))),0.3),0.02));//本部分在实际使用时max(min((0.131*36.0/(timerate)*(omega(3.*0.00000465,0.00000465))/(omega(3.*Rs,Rs))),0.3),0.02)由uniform输入
    blendWeight = (iFrame<2 || iMouse.z > 0.0 ) ? 1.0 : blendWeight;
    
    vec4 previousColor = texelFetch(iChannel3, ivec2(fragCoord), 0); //获取前一帧的颜色
    fragColor = (blendWeight)*fragColor+(1.0-blendWeight)*previousColor; //混合当前帧和前一帧
       
       //uv=DirTouv();
        
       //fragColor=texelFetch(iChannel1, ivec2(uv*iChannelResolution[1].xy), 0 );
       //fragColor=vec4(0.1*log(fragColor.r+1.),0.1*log(fragColor.g+1.),0.1*log(fragColor.b+1.),0.1*log(fragColor.a+1.));

}
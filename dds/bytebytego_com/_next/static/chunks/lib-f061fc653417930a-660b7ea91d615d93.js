try{let e="undefined"!=typeof window?window:"undefined"!=typeof global?global:"undefined"!=typeof globalThis?globalThis:"undefined"!=typeof self?self:{},t=(new e.Error).stack;t&&(e._sentryDebugIds=e._sentryDebugIds||{},e._sentryDebugIds[t]="5cf34663-f3ac-4c50-9d13-e8ec7372507e",e._sentryDebugIdIdentifier="sentry-dbid-5cf34663-f3ac-4c50-9d13-e8ec7372507e")}catch(e){}"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[5638],{94302:(e,t,n)=>{n.d(t,{U$:()=>eJ,aD:()=>$,e4:()=>N,j0:()=>eY});var r=n(78765),o=n(44574),i=n(20277),a=n(96306),l=n(98160),u=(n(43350),n(65873));let c=r.O,s=c.document,_=c.navigator,d="Report a Bug",f="Cancel",p="Send Bug Report",h="Confirm",g="Report a Bug",m="your.email@example.org",v="Email",b="What's the bug? What did you expect?",y="Description",x="Your Name",w="Name",k="Thank you for your report!",C="(required)",S="Add a screenshot",E="Remove screenshot",H=(e,t={includeReplay:!0})=>{if(!e.message)throw Error("Unable to submit feedback with empty message");let n=(0,o.KU)();if(!n)throw Error("No client setup, cannot send feedback.");e.tags&&Object.keys(e.tags).length&&(0,o.o5)().setTags(e.tags);let r=(0,i.q)({source:"api",url:(0,a.$N)(),...e},t);return new Promise((e,t)=>{let o=setTimeout(()=>t("Unable to determine if Feedback was correctly sent."),5e3),i=n.on("afterSendEvent",(n,a)=>{if(n.event_id===r)return(clearTimeout(o),i(),a&&"number"==typeof a.statusCode&&a.statusCode>=200&&a.statusCode<300)?e(r):a&&"number"==typeof a.statusCode&&0===a.statusCode?t("Unable to send Feedback. This is because of network issues, or because you are using an ad-blocker."):a&&"number"==typeof a.statusCode&&403===a.statusCode?t("Unable to send Feedback. This could be because this domain is not in your list of allowed domains."):t("Unable to send Feedback. This could be because of network issues, or because you are using an ad-blocker")})})};function F(e,t){return{...e,...t,tags:{...e.tags,...t.tags},onFormOpen:()=>{t.onFormOpen?.(),e.onFormOpen?.()},onFormClose:()=>{t.onFormClose?.(),e.onFormClose?.()},onSubmitSuccess:n=>{t.onSubmitSuccess?.(n),e.onSubmitSuccess?.(n)},onSubmitError:n=>{t.onSubmitError?.(n),e.onSubmitError?.(n)},onFormSubmitted:()=>{t.onFormSubmitted?.(),e.onFormSubmitted?.()},themeDark:{...e.themeDark,...t.themeDark},themeLight:{...e.themeLight,...t.themeLight}}}function L(e,t){return Object.entries(t).forEach(([t,n])=>{e.setAttributeNS(null,t,n)}),e}let T="rgba(88, 74, 192, 1)",D={foreground:"#2b2233",background:"#ffffff",accentForeground:"white",accentBackground:T,successColor:"#268d75",errorColor:"#df3338",border:"1.5px solid rgba(41, 35, 47, 0.13)",boxShadow:"0px 4px 24px 0px rgba(43, 34, 51, 0.12)",outline:"1px auto var(--accent-background)",interactiveFilter:"brightness(95%)"},M={foreground:"#ebe6ef",background:"#29232f",accentForeground:"white",accentBackground:T,successColor:"#2da98c",errorColor:"#f55459",border:"1.5px solid rgba(235, 230, 239, 0.15)",boxShadow:"0px 4px 24px 0px rgba(43, 34, 51, 0.12)",outline:"1px auto var(--accent-background)",interactiveFilter:"brightness(150%)"};function P(e){return`
  --foreground: ${e.foreground};
  --background: ${e.background};
  --accent-foreground: ${e.accentForeground};
  --accent-background: ${e.accentBackground};
  --success-color: ${e.successColor};
  --error-color: ${e.errorColor};
  --border: ${e.border};
  --box-shadow: ${e.boxShadow};
  --outline: ${e.outline};
  --interactive-filter: ${e.interactiveFilter};
  `}let $=({lazyLoadIntegration:e,getModalIntegration:t,getScreenshotIntegration:n})=>({id:r="sentry-feedback",autoInject:o=!0,showBranding:i=!0,isEmailRequired:a=!1,isNameRequired:T=!1,showEmail:$=!0,showName:N=!0,enableScreenshot:A=!0,useSentryUser:R={email:"email",name:"username"},tags:U,styleNonce:B,scriptNonce:z,colorScheme:V="system",themeLight:q={},themeDark:I={},addScreenshotButtonLabel:W=S,cancelButtonLabel:O=f,confirmButtonLabel:j=h,emailLabel:Z=v,emailPlaceholder:K=m,formTitle:Y=g,isRequiredLabel:G=C,messageLabel:Q=y,messagePlaceholder:X=b,nameLabel:J=w,namePlaceholder:ee=x,removeScreenshotButtonLabel:et=E,submitButtonLabel:en=p,successMessageText:er=k,triggerLabel:eo=d,triggerAriaLabel:ei="",onFormOpen:ea,onFormClose:el,onSubmitSuccess:eu,onSubmitError:ec,onFormSubmitted:es}={})=>{let e_={id:r,autoInject:o,showBranding:i,isEmailRequired:a,isNameRequired:T,showEmail:$,showName:N,enableScreenshot:A,useSentryUser:R,tags:U,styleNonce:B,scriptNonce:z,colorScheme:V,themeDark:I,themeLight:q,triggerLabel:eo,triggerAriaLabel:ei,cancelButtonLabel:O,submitButtonLabel:en,confirmButtonLabel:j,formTitle:Y,emailLabel:Z,emailPlaceholder:K,messageLabel:Q,messagePlaceholder:X,nameLabel:J,namePlaceholder:ee,successMessageText:er,isRequiredLabel:G,addScreenshotButtonLabel:W,removeScreenshotButtonLabel:et,onFormClose:el,onFormOpen:ea,onSubmitError:ec,onSubmitSuccess:eu,onFormSubmitted:es},ed=null,ef=[],ep=e=>{if(!ed){let t=s.createElement("div");t.id=String(e.id),s.body.appendChild(t),(ed=t.attachShadow({mode:"open"})).appendChild(function({colorScheme:e,themeDark:t,themeLight:n,styleNonce:r}){let o=s.createElement("style");return o.textContent=`
:host {
  --font-family: system-ui, 'Helvetica Neue', Arial, sans-serif;
  --font-size: 14px;
  --z-index: 100000;

  --page-margin: 16px;
  --inset: auto 0 0 auto;
  --actor-inset: var(--inset);

  font-family: var(--font-family);
  font-size: var(--font-size);

  ${"system"!==e?"color-scheme: only light;":""}

  ${P("dark"===e?{...M,...t}:{...D,...n})}
}

${"system"===e?`
@media (prefers-color-scheme: dark) {
  :host {
    ${P({...M,...t})}
  }
}`:""}
}
`,r&&o.setAttribute("nonce",r),o}(e))}return ed},eh=async r=>{let o,i,a=r.enableScreenshot&&!(/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(_.userAgent)||/Macintosh/i.test(_.userAgent)&&_.maxTouchPoints&&_.maxTouchPoints>1)&&!!isSecureContext;try{o=(t?t():await e("feedbackModalIntegration",z))(),(0,l.Q8)(o)}catch{throw Error("[Feedback] Missing feedback modal integration!")}try{let t=a?n?n():await e("feedbackScreenshotIntegration",z):void 0;t&&(i=t(),(0,l.Q8)(i))}catch{}let u=o.createDialog({options:{...r,onFormClose:()=>{u?.close(),r.onFormClose?.()},onFormSubmitted:()=>{u?.close(),r.onFormSubmitted?.()}},screenshotIntegration:i,sendFeedback:H,shadow:ep(r)});return u},eg=(e,t={})=>{let n=F(e_,t),r="string"==typeof e?s.querySelector(e):"function"==typeof e.addEventListener?e:null;if(!r)throw Error("Unable to attach to target element");let o=null,i=async()=>{o||(o=await eh({...n,onFormSubmitted:()=>{o?.removeFromDom(),n.onFormSubmitted?.()}})),o.appendToDom(),o.open()};r.addEventListener("click",i);let a=()=>{ef=ef.filter(e=>e!==a),o?.removeFromDom(),o=null,r.removeEventListener("click",i)};return ef.push(a),a},em=(e={})=>{let t=F(e_,e),n=ep(t),r=function({triggerLabel:e,triggerAriaLabel:t,shadow:n,styleNonce:r}){let o=s.createElement("button");if(o.type="button",o.className="widget__actor",o.ariaHidden="false",o.ariaLabel=t||e||d,o.appendChild(function(){let e=e=>c.document.createElementNS("http://www.w3.org/2000/svg",e),t=L(e("svg"),{width:"20",height:"20",viewBox:"0 0 20 20",fill:"var(--actor-color, var(--foreground))"}),n=L(e("g"),{clipPath:"url(#clip0_57_80)"}),r=L(e("path"),{"fill-rule":"evenodd","clip-rule":"evenodd",d:"M15.6622 15H12.3997C12.2129 14.9959 12.031 14.9396 11.8747 14.8375L8.04965 12.2H7.49956V19.1C7.4875 19.3348 7.3888 19.5568 7.22256 19.723C7.05632 19.8892 6.83435 19.9879 6.59956 20H2.04956C1.80193 19.9968 1.56535 19.8969 1.39023 19.7218C1.21511 19.5467 1.1153 19.3101 1.11206 19.0625V12.2H0.949652C0.824431 12.2017 0.700142 12.1783 0.584123 12.1311C0.468104 12.084 0.362708 12.014 0.274155 11.9255C0.185602 11.8369 0.115689 11.7315 0.0685419 11.6155C0.0213952 11.4995 -0.00202913 11.3752 -0.00034808 11.25V3.75C-0.00900498 3.62067 0.0092504 3.49095 0.0532651 3.36904C0.0972798 3.24712 0.166097 3.13566 0.255372 3.04168C0.344646 2.94771 0.452437 2.87327 0.571937 2.82307C0.691437 2.77286 0.82005 2.74798 0.949652 2.75H8.04965L11.8747 0.1625C12.031 0.0603649 12.2129 0.00407221 12.3997 0H15.6622C15.9098 0.00323746 16.1464 0.103049 16.3215 0.278167C16.4966 0.453286 16.5964 0.689866 16.5997 0.9375V3.25269C17.3969 3.42959 18.1345 3.83026 18.7211 4.41679C19.5322 5.22788 19.9878 6.32796 19.9878 7.47502C19.9878 8.62209 19.5322 9.72217 18.7211 10.5333C18.1345 11.1198 17.3969 11.5205 16.5997 11.6974V14.0125C16.6047 14.1393 16.5842 14.2659 16.5395 14.3847C16.4948 14.5035 16.4268 14.6121 16.3394 14.7042C16.252 14.7962 16.147 14.8698 16.0307 14.9206C15.9144 14.9714 15.7891 14.9984 15.6622 15ZM1.89695 10.325H1.88715V4.625H8.33715C8.52423 4.62301 8.70666 4.56654 8.86215 4.4625L12.6872 1.875H14.7247V13.125H12.6872L8.86215 10.4875C8.70666 10.3835 8.52423 10.327 8.33715 10.325H2.20217C2.15205 10.3167 2.10102 10.3125 2.04956 10.3125C1.9981 10.3125 1.94708 10.3167 1.89695 10.325ZM2.98706 12.2V18.1625H5.66206V12.2H2.98706ZM16.5997 9.93612V5.01393C16.6536 5.02355 16.7072 5.03495 16.7605 5.04814C17.1202 5.13709 17.4556 5.30487 17.7425 5.53934C18.0293 5.77381 18.2605 6.06912 18.4192 6.40389C18.578 6.73866 18.6603 7.10452 18.6603 7.47502C18.6603 7.84552 18.578 8.21139 18.4192 8.54616C18.2605 8.88093 18.0293 9.17624 17.7425 9.41071C17.4556 9.64518 17.1202 9.81296 16.7605 9.90191C16.7072 9.91509 16.6536 9.9265 16.5997 9.93612Z"});t.appendChild(n).appendChild(r);let o=e("defs"),i=L(e("clipPath"),{id:"clip0_57_80"}),a=L(e("rect"),{width:"20",height:"20",fill:"white"});return i.appendChild(a),o.appendChild(i),t.appendChild(o).appendChild(i).appendChild(a),t}()),e){let t=s.createElement("span");t.appendChild(s.createTextNode(e)),o.appendChild(t)}let i=function(e){let t=s.createElement("style");return t.textContent=`
.widget__actor {
  position: fixed;
  z-index: var(--z-index);
  margin: var(--page-margin);
  inset: var(--actor-inset);

  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px;

  font-family: inherit;
  font-size: var(--font-size);
  font-weight: 600;
  line-height: 1.14em;
  text-decoration: none;

  background: var(--actor-background, var(--background));
  border-radius: var(--actor-border-radius, 1.7em/50%);
  border: var(--actor-border, var(--border));
  box-shadow: var(--actor-box-shadow, var(--box-shadow));
  color: var(--actor-color, var(--foreground));
  fill: var(--actor-color, var(--foreground));
  cursor: pointer;
  opacity: 1;
  transition: transform 0.2s ease-in-out;
  transform: translate(0, 0) scale(1);
}
.widget__actor[aria-hidden="true"] {
  opacity: 0;
  pointer-events: none;
  visibility: hidden;
  transform: translate(0, 16px) scale(0.98);
}

.widget__actor:hover {
  background: var(--actor-hover-background, var(--background));
  filter: var(--interactive-filter);
}

.widget__actor svg {
  width: 1.14em;
  height: 1.14em;
}

@media (max-width: 600px) {
  .widget__actor span {
    display: none;
  }
}
`,e&&t.setAttribute("nonce",e),t}(r);return{el:o,appendToDom(){n.appendChild(i),n.appendChild(o)},removeFromDom(){o.remove(),i.remove()},show(){o.ariaHidden="false"},hide(){o.ariaHidden="true"}}}({triggerLabel:t.triggerLabel,triggerAriaLabel:t.triggerAriaLabel,shadow:n,styleNonce:B});return eg(r.el,{...t,onFormOpen(){r.hide()},onFormClose(){r.show()},onFormSubmitted(){r.show()}}),r};return{name:"Feedback",setupOnce(){(0,u.B)()&&e_.autoInject&&("loading"===s.readyState?s.addEventListener("DOMContentLoaded",()=>em().appendToDom()):em().appendToDom())},attachTo:eg,createWidget(e={}){let t=em(F(e_,e));return t.appendToDom(),t},createForm:async(e={})=>eh(F(e_,e)),remove(){ed&&(ed.parentElement?.remove(),ed=null),ef.forEach(e=>e()),ef=[]}}};function N(){let e=(0,o.KU)();return e?.getIntegrationByName("Feedback")}var A,R,U,B,z,V,q,I={},W=[],O=/acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i,j=Array.isArray;function Z(e,t){for(var n in t)e[n]=t[n];return e}function K(e){var t=e.parentNode;t&&t.removeChild(e)}function Y(e,t,n){var r,o,i,a={};for(i in t)"key"==i?r=t[i]:"ref"==i?o=t[i]:a[i]=t[i];if(arguments.length>2&&(a.children=arguments.length>3?A.call(arguments,2):n),"function"==typeof e&&null!=e.defaultProps)for(i in e.defaultProps)void 0===a[i]&&(a[i]=e.defaultProps[i]);return G(e,a,r,o,null)}function G(e,t,n,r,o){var i={type:e,props:t,key:n,ref:r,__k:null,__:null,__b:0,__e:null,__d:void 0,__c:null,constructor:void 0,__v:null==o?++U:o,__i:-1,__u:0};return null==o&&null!=R.vnode&&R.vnode(i),i}function Q(e){return e.children}function X(e,t){this.props=e,this.context=t}function J(e,t){if(null==t)return e.__?J(e.__,e.__i+1):null;for(var n;t<e.__k.length;t++)if(null!=(n=e.__k[t])&&null!=n.__e)return n.__e;return"function"==typeof e.type?J(e):null}function ee(e){(!e.__d&&(e.__d=!0)&&B.push(e)&&!et.__r++||z!==R.debounceRendering)&&((z=R.debounceRendering)||V)(et)}function et(){var e,t,n,r=[],o=[];for(B.sort(q);e=B.shift();)e.__d&&(n=B.length,t=function(e,t,n){var r,o=e.__v,i=o.__e,a=e.__P;if(a)return(r=Z({},o)).__v=o.__v+1,R.vnode&&R.vnode(r),el(a,r,o,e.__n,void 0!==a.ownerSVGElement,32&o.__u?[i]:null,t,null==i?J(o):i,!!(32&o.__u),n),r.__.__k[r.__i]=r,r.__d=void 0,r.__e!=i&&function e(t){var n,r;if(null!=(t=t.__)&&null!=t.__c){for(t.__e=t.__c.base=null,n=0;n<t.__k.length;n++)if(null!=(r=t.__k[n])&&null!=r.__e){t.__e=t.__c.base=r.__e;break}return e(t)}}(r),r}(e,r,o)||t,0===n||B.length>n?(eu(r,t,o),o.length=r.length=0,t=void 0,B.sort(q)):t&&R.__c&&R.__c(t,W));t&&eu(r,t,o),et.__r=0}function en(e,t,n,r,o,i,a,l,u,c,s){var _,d,f,p,h,g=r&&r.__k||W,m=t.length;for(n.__d=u,function(e,t,n){var r,o,i,a,l,u=t.length,c=n.length,s=c,_=0;for(e.__k=[],r=0;r<u;r++)null!=(o=e.__k[r]=null==(o=t[r])||"boolean"==typeof o||"function"==typeof o?null:"string"==typeof o||"number"==typeof o||"bigint"==typeof o||o.constructor==String?G(null,o,null,null,o):j(o)?G(Q,{children:o},null,null,null):void 0===o.constructor&&o.__b>0?G(o.type,o.props,o.key,o.ref?o.ref:null,o.__v):o)?(o.__=e,o.__b=e.__b+1,l=function(e,t,n,r){var o=e.key,i=e.type,a=n-1,l=n+1,u=t[n];if(null===u||u&&o==u.key&&i===u.type)return n;if(r>+(null!=u&&0==(131072&u.__u)))for(;a>=0||l<t.length;){if(a>=0){if((u=t[a])&&0==(131072&u.__u)&&o==u.key&&i===u.type)return a;a--}if(l<t.length){if((u=t[l])&&0==(131072&u.__u)&&o==u.key&&i===u.type)return l;l++}}return -1}(o,n,a=r+_,s),o.__i=l,i=null,-1!==l&&(s--,(i=n[l])&&(i.__u|=131072)),null==i||null===i.__v?(-1==l&&_--,"function"!=typeof o.type&&(o.__u|=65536)):l!==a&&(l===a+1?_++:l>a?s>u-a?_+=l-a:_--:_=l<a&&l==a-1?l-a:0,l!==r+_&&(o.__u|=65536))):(i=n[r])&&null==i.key&&i.__e&&(i.__e==e.__d&&(e.__d=J(i)),es(i,i,!1),n[r]=null,s--);if(s)for(r=0;r<c;r++)null!=(i=n[r])&&0==(131072&i.__u)&&(i.__e==e.__d&&(e.__d=J(i)),es(i,i))}(n,t,g),u=n.__d,_=0;_<m;_++)null!=(f=n.__k[_])&&"boolean"!=typeof f&&"function"!=typeof f&&(d=-1===f.__i?I:g[f.__i]||I,f.__i=_,el(e,f,d,o,i,a,l,u,c,s),p=f.__e,f.ref&&d.ref!=f.ref&&(d.ref&&ec(d.ref,null,f),s.push(f.ref,f.__c||p,f)),null==h&&null!=p&&(h=p),65536&f.__u||d.__k===f.__k?u=function e(t,n,r){var o,i;if("function"==typeof t.type){for(o=t.__k,i=0;o&&i<o.length;i++)o[i]&&(o[i].__=t,n=e(o[i],n,r));return n}t.__e!=n&&(r.insertBefore(t.__e,n||null),n=t.__e);do n=n&&n.nextSibling;while(null!=n&&8===n.nodeType);return n}(f,u,e):"function"==typeof f.type&&void 0!==f.__d?u=f.__d:p&&(u=p.nextSibling),f.__d=void 0,f.__u&=-196609);n.__d=u,n.__e=h}function er(e,t,n){"-"===t[0]?e.setProperty(t,null==n?"":n):e[t]=null==n?"":"number"!=typeof n||O.test(t)?n:n+"px"}function eo(e,t,n,r,o){var i;e:if("style"===t)if("string"==typeof n)e.style.cssText=n;else{if("string"==typeof r&&(e.style.cssText=r=""),r)for(t in r)n&&t in n||er(e.style,t,"");if(n)for(t in n)r&&n[t]===r[t]||er(e.style,t,n[t])}else if("o"===t[0]&&"n"===t[1])i=t!==(t=t.replace(/(PointerCapture)$|Capture$/i,"$1")),t=t.toLowerCase()in e?t.toLowerCase().slice(2):t.slice(2),e.l||(e.l={}),e.l[t+i]=n,n?r?n.u=r.u:(n.u=Date.now(),e.addEventListener(t,i?ea:ei,i)):e.removeEventListener(t,i?ea:ei,i);else{if(o)t=t.replace(/xlink(H|:h)/,"h").replace(/sName$/,"s");else if("width"!==t&&"height"!==t&&"href"!==t&&"list"!==t&&"form"!==t&&"tabIndex"!==t&&"download"!==t&&"rowSpan"!==t&&"colSpan"!==t&&"role"!==t&&t in e)try{e[t]=null==n?"":n;break e}catch(e){}"function"==typeof n||(null==n||!1===n&&"-"!==t[4]?e.removeAttribute(t):e.setAttribute(t,n))}}function ei(e){if(this.l){var t=this.l[e.type+!1];if(e.t){if(e.t<=t.u)return}else e.t=Date.now();return t(R.event?R.event(e):e)}}function ea(e){if(this.l)return this.l[e.type+!0](R.event?R.event(e):e)}function el(e,t,n,r,o,i,a,l,u,c){var s,_,d,f,p,h,g,m,v,b,y,x,w,k,C,S=t.type;if(void 0!==t.constructor)return null;128&n.__u&&(u=!!(32&n.__u),i=[l=t.__e=n.__e]),(s=R.__b)&&s(t);e:if("function"==typeof S)try{if(m=t.props,v=(s=S.contextType)&&r[s.__c],b=s?v?v.props.value:s.__:r,n.__c?g=(_=t.__c=n.__c).__=_.__E:("prototype"in S&&S.prototype.render?t.__c=_=new S(m,b):(t.__c=_=new X(m,b),_.constructor=S,_.render=e_),v&&v.sub(_),_.props=m,_.state||(_.state={}),_.context=b,_.__n=r,d=_.__d=!0,_.__h=[],_._sb=[]),null==_.__s&&(_.__s=_.state),null!=S.getDerivedStateFromProps&&(_.__s==_.state&&(_.__s=Z({},_.__s)),Z(_.__s,S.getDerivedStateFromProps(m,_.__s))),f=_.props,p=_.state,_.__v=t,d)null==S.getDerivedStateFromProps&&null!=_.componentWillMount&&_.componentWillMount(),null!=_.componentDidMount&&_.__h.push(_.componentDidMount);else{if(null==S.getDerivedStateFromProps&&m!==f&&null!=_.componentWillReceiveProps&&_.componentWillReceiveProps(m,b),!_.__e&&(null!=_.shouldComponentUpdate&&!1===_.shouldComponentUpdate(m,_.__s,b)||t.__v===n.__v)){for(t.__v!==n.__v&&(_.props=m,_.state=_.__s,_.__d=!1),t.__e=n.__e,t.__k=n.__k,t.__k.forEach(function(e){e&&(e.__=t)}),y=0;y<_._sb.length;y++)_.__h.push(_._sb[y]);_._sb=[],_.__h.length&&a.push(_);break e}null!=_.componentWillUpdate&&_.componentWillUpdate(m,_.__s,b),null!=_.componentDidUpdate&&_.__h.push(function(){_.componentDidUpdate(f,p,h)})}if(_.context=b,_.props=m,_.__P=e,_.__e=!1,x=R.__r,w=0,"prototype"in S&&S.prototype.render){for(_.state=_.__s,_.__d=!1,x&&x(t),s=_.render(_.props,_.state,_.context),k=0;k<_._sb.length;k++)_.__h.push(_._sb[k]);_._sb=[]}else do _.__d=!1,x&&x(t),s=_.render(_.props,_.state,_.context),_.state=_.__s;while(_.__d&&++w<25);_.state=_.__s,null!=_.getChildContext&&(r=Z(Z({},r),_.getChildContext())),d||null==_.getSnapshotBeforeUpdate||(h=_.getSnapshotBeforeUpdate(f,p)),en(e,j(C=null!=s&&s.type===Q&&null==s.key?s.props.children:s)?C:[C],t,n,r,o,i,a,l,u,c),_.base=t.__e,t.__u&=-161,_.__h.length&&a.push(_),g&&(_.__E=_.__=null)}catch(e){t.__v=null,u||null!=i?(t.__e=l,t.__u|=u?160:32,i[i.indexOf(l)]=null):(t.__e=n.__e,t.__k=n.__k),R.__e(e,t,n)}else null==i&&t.__v===n.__v?(t.__k=n.__k,t.__e=n.__e):t.__e=function(e,t,n,r,o,i,a,l,u){var c,s,_,d,f,p,h,g=n.props,m=t.props,v=t.type;if("svg"===v&&(o=!0),null!=i){for(c=0;c<i.length;c++)if((f=i[c])&&"setAttribute"in f==!!v&&(v?f.localName===v:3===f.nodeType)){e=f,i[c]=null;break}}if(null==e){if(null===v)return document.createTextNode(m);e=o?document.createElementNS("http://www.w3.org/2000/svg",v):document.createElement(v,m.is&&m),i=null,l=!1}if(null===v)g===m||l&&e.data===m||(e.data=m);else{if(i=i&&A.call(e.childNodes),g=n.props||I,!l&&null!=i)for(g={},c=0;c<e.attributes.length;c++)g[(f=e.attributes[c]).name]=f.value;for(c in g)f=g[c],"children"==c||("dangerouslySetInnerHTML"==c?_=f:"key"===c||c in m||eo(e,c,null,f,o));for(c in m)f=m[c],"children"==c?d=f:"dangerouslySetInnerHTML"==c?s=f:"value"==c?p=f:"checked"==c?h=f:"key"===c||l&&"function"!=typeof f||g[c]===f||eo(e,c,f,g[c],o);if(s)l||_&&(s.__html===_.__html||s.__html===e.innerHTML)||(e.innerHTML=s.__html),t.__k=[];else if(_&&(e.innerHTML=""),en(e,j(d)?d:[d],t,n,r,o&&"foreignObject"!==v,i,a,i?i[0]:n.__k&&J(n,0),l,u),null!=i)for(c=i.length;c--;)null!=i[c]&&K(i[c]);l||(c="value",void 0===p||p===e[c]&&("progress"!==v||p)&&("option"!==v||p===g[c])||eo(e,c,p,g[c],!1),c="checked",void 0!==h&&h!==e[c]&&eo(e,c,h,g[c],!1))}return e}(n.__e,t,n,r,o,i,a,u,c);(s=R.diffed)&&s(t)}function eu(e,t,n){for(var r=0;r<n.length;r++)ec(n[r],n[++r],n[++r]);R.__c&&R.__c(t,e),e.some(function(t){try{e=t.__h,t.__h=[],e.some(function(e){e.call(t)})}catch(e){R.__e(e,t.__v)}})}function ec(e,t,n){try{"function"==typeof e?e(t):e.current=t}catch(e){R.__e(e,n)}}function es(e,t,n){var r,o;if(R.unmount&&R.unmount(e),(r=e.ref)&&(r.current&&r.current!==e.__e||ec(r,null,t)),null!=(r=e.__c)){if(r.componentWillUnmount)try{r.componentWillUnmount()}catch(e){R.__e(e,t)}r.base=r.__P=null,e.__c=void 0}if(r=e.__k)for(o=0;o<r.length;o++)r[o]&&es(r[o],t,n||"function"!=typeof e.type);n||null==e.__e||K(e.__e),e.__=e.__e=e.__d=void 0}function e_(e,t,n){return this.constructor(e,n)}A=W.slice,R={__e:function(e,t,n,r){for(var o,i,a;t=t.__;)if((o=t.__c)&&!o.__)try{if((i=o.constructor)&&null!=i.getDerivedStateFromError&&(o.setState(i.getDerivedStateFromError(e)),a=o.__d),null!=o.componentDidCatch&&(o.componentDidCatch(e,r||{}),a=o.__d),a)return o.__E=o}catch(t){e=t}throw e}},U=0,X.prototype.setState=function(e,t){var n;n=null!=this.__s&&this.__s!==this.state?this.__s:this.__s=Z({},this.state),"function"==typeof e&&(e=e(Z({},n),this.props)),e&&Z(n,e),null!=e&&this.__v&&(t&&this._sb.push(t),ee(this))},X.prototype.forceUpdate=function(e){this.__v&&(this.__e=!0,e&&this.__h.push(e),ee(this))},X.prototype.render=Q,B=[],V="function"==typeof Promise?Promise.prototype.then.bind(Promise.resolve()):setTimeout,q=function(e,t){return e.__v.__b-t.__v.__b},et.__r=0;var ed,ef,ep,eh,eg=0,em=[],ev=[],eb=R,ey=eb.__b,ex=eb.__r,ew=eb.diffed,ek=eb.__c,eC=eb.unmount,eS=eb.__;function eE(e,t){eb.__h&&eb.__h(ef,e,eg||t),eg=0;var n=ef.__H||(ef.__H={__:[],__h:[]});return e>=n.__.length&&n.__.push({__V:ev}),n.__[e]}function eH(e){return eg=1,eF(eR,e)}function eF(e,t,n){var r=eE(ed++,2);if(r.t=e,!r.__c&&(r.__=[n?n(t):eR(void 0,t),function(e){var t=r.__N?r.__N[0]:r.__[0],n=r.t(t,e);t!==n&&(r.__N=[n,r.__[1]],r.__c.setState({}))}],r.__c=ef,!ef.u)){var o=function(e,t,n){if(!r.__c.__H)return!0;var o=r.__c.__H.__.filter(function(e){return!!e.__c});if(o.every(function(e){return!e.__N}))return!i||i.call(this,e,t,n);var a=!1;return o.forEach(function(e){if(e.__N){var t=e.__[0];e.__=e.__N,e.__N=void 0,t!==e.__[0]&&(a=!0)}}),!(!a&&r.__c.props===e)&&(!i||i.call(this,e,t,n))};ef.u=!0;var i=ef.shouldComponentUpdate,a=ef.componentWillUpdate;ef.componentWillUpdate=function(e,t,n){if(this.__e){var r=i;i=void 0,o(e,t,n),i=r}a&&a.call(this,e,t,n)},ef.shouldComponentUpdate=o}return r.__N||r.__}function eL(e,t){var n=eE(ed++,4);!eb.__s&&eA(n.__H,t)&&(n.__=e,n.i=t,ef.__h.push(n))}function eT(e,t){var n=eE(ed++,7);return eA(n.__H,t)?(n.__V=e(),n.i=t,n.__h=e,n.__V):n.__}function eD(e,t){return eg=8,eT(function(){return e},t)}function eM(){for(var e;e=em.shift();)if(e.__P&&e.__H)try{e.__H.__h.forEach(e$),e.__H.__h.forEach(eN),e.__H.__h=[]}catch(t){e.__H.__h=[],eb.__e(t,e.__v)}}eb.__b=function(e){ef=null,ey&&ey(e)},eb.__=function(e,t){t.__k&&t.__k.__m&&(e.__m=t.__k.__m),eS&&eS(e,t)},eb.__r=function(e){ex&&ex(e),ed=0;var t=(ef=e.__c).__H;t&&(ep===ef?(t.__h=[],ef.__h=[],t.__.forEach(function(e){e.__N&&(e.__=e.__N),e.__V=ev,e.__N=e.i=void 0})):(t.__h.forEach(e$),t.__h.forEach(eN),t.__h=[],ed=0)),ep=ef},eb.diffed=function(e){ew&&ew(e);var t=e.__c;t&&t.__H&&(t.__H.__h.length&&(1!==em.push(t)&&eh===eb.requestAnimationFrame||((eh=eb.requestAnimationFrame)||function(e){var t,n=function(){clearTimeout(r),eP&&cancelAnimationFrame(t),setTimeout(e)},r=setTimeout(n,100);eP&&(t=requestAnimationFrame(n))})(eM)),t.__H.__.forEach(function(e){e.i&&(e.__H=e.i),e.__V!==ev&&(e.__=e.__V),e.i=void 0,e.__V=ev})),ep=ef=null},eb.__c=function(e,t){t.some(function(e){try{e.__h.forEach(e$),e.__h=e.__h.filter(function(e){return!e.__||eN(e)})}catch(n){t.some(function(e){e.__h&&(e.__h=[])}),t=[],eb.__e(n,e.__v)}}),ek&&ek(e,t)},eb.unmount=function(e){eC&&eC(e);var t,n=e.__c;n&&n.__H&&(n.__H.__.forEach(function(e){try{e$(e)}catch(e){t=e}}),n.__H=void 0,t&&eb.__e(t,n.__v))};var eP="function"==typeof requestAnimationFrame;function e$(e){var t=ef,n=e.__c;"function"==typeof n&&(e.__c=void 0,n()),ef=t}function eN(e){var t=ef;e.__c=e.__(),ef=t}function eA(e,t){return!e||e.length!==t.length||t.some(function(t,n){return t!==e[n]})}function eR(e,t){return"function"==typeof t?t(e):t}let eU=Object.defineProperty({__proto__:null,useCallback:eD,useContext:function(e){var t=ef.context[e.__c],n=eE(ed++,9);return n.c=e,t?(null==n.__&&(n.__=!0,t.sub(ef)),t.props.value):e.__},useDebugValue:function(e,t){eb.useDebugValue&&eb.useDebugValue(t?t(e):e)},useEffect:function(e,t){var n=eE(ed++,3);!eb.__s&&eA(n.__H,t)&&(n.__=e,n.i=t,ef.__H.__h.push(n))},useErrorBoundary:function(e){var t=eE(ed++,10),n=eH();return t.__=e,ef.componentDidCatch||(ef.componentDidCatch=function(e,r){t.__&&t.__(e,r),n[1](e)}),[n[0],function(){n[1](void 0)}]},useId:function(){var e=eE(ed++,11);if(!e.__){for(var t=ef.__v;null!==t&&!t.__m&&null!==t.__;)t=t.__;var n=t.__m||(t.__m=[0,0]);e.__="P"+n[0]+"-"+n[1]++}return e.__},useImperativeHandle:function(e,t,n){eg=6,eL(function(){return"function"==typeof e?(e(t()),function(){return e(null)}):e?(e.current=t(),function(){return e.current=null}):void 0},null==n?n:n.concat(e))},useLayoutEffect:eL,useMemo:eT,useReducer:eF,useRef:function(e){return eg=5,eT(function(){return{current:e}},[])},useState:eH},Symbol.toStringTag,{value:"Module"});function eB({options:e}){let t=eT(()=>({__html:function(){let e=e=>s.createElementNS("http://www.w3.org/2000/svg",e),t=L(e("svg"),{width:"32",height:"30",viewBox:"0 0 72 66",fill:"inherit"}),n=L(e("path"),{transform:"translate(11, 11)",d:"M29,2.26a4.67,4.67,0,0,0-8,0L14.42,13.53A32.21,32.21,0,0,1,32.17,40.19H27.55A27.68,27.68,0,0,0,12.09,17.47L6,28a15.92,15.92,0,0,1,9.23,12.17H4.62A.76.76,0,0,1,4,39.06l2.94-5a10.74,10.74,0,0,0-3.36-1.9l-2.91,5a4.54,4.54,0,0,0,1.69,6.24A4.66,4.66,0,0,0,4.62,44H19.15a19.4,19.4,0,0,0-8-17.31l2.31-4A23.87,23.87,0,0,1,23.76,44H36.07a35.88,35.88,0,0,0-16.41-31.8l4.67-8a.77.77,0,0,1,1.05-.27c.53.29,20.29,34.77,20.66,35.17a.76.76,0,0,1-.68,1.13H40.6q.09,1.91,0,3.81h4.78A4.59,4.59,0,0,0,50,39.43a4.49,4.49,0,0,0-.62-2.28Z"});return t.appendChild(n),t}().outerHTML}),[]);return Y("h2",{class:"dialog__header"},Y("span",{class:"dialog__title"},e.formTitle),e.showBranding?Y("a",{class:"brand-link",target:"_blank",href:"https://sentry.io/welcome/",title:"Powered by Sentry",rel:"noopener noreferrer",dangerouslySetInnerHTML:t}):null)}function ez(e,t){let n=e.get(t);return"string"==typeof n?n.trim():""}function eV({options:e,defaultEmail:t,defaultName:n,onFormClose:r,onSubmit:o,onSubmitSuccess:i,onSubmitError:a,showEmail:l,showName:u,screenshotInput:c}){let{tags:s,addScreenshotButtonLabel:_,removeScreenshotButtonLabel:d,cancelButtonLabel:f,emailLabel:p,emailPlaceholder:h,isEmailRequired:g,isNameRequired:m,messageLabel:v,messagePlaceholder:b,nameLabel:y,namePlaceholder:x,submitButtonLabel:w,isRequiredLabel:k}=e,[C,S]=eH(!1),[E,H]=eH(null),[F,L]=eH(!1),T=c?.input,[D,M]=eH(null),P=eD(e=>{M(e),L(!1)},[]),$=eD(e=>{let t=function(e,t){let n=[];return t.isNameRequired&&!e.name&&n.push(t.nameLabel),t.isEmailRequired&&!e.email&&n.push(t.emailLabel),e.message||n.push(t.messageLabel),n}(e,{emailLabel:p,isEmailRequired:g,isNameRequired:m,messageLabel:v,nameLabel:y});return t.length>0?H(`Please enter in the following required fields: ${t.join(", ")}`):H(null),0===t.length},[p,g,m,v,y]);return Y("form",{class:"form",onSubmit:eD(async e=>{S(!0);try{if(e.preventDefault(),!(e.target instanceof HTMLFormElement))return;let t=new FormData(e.target),n=await (c&&F?c.value():void 0),r={name:ez(t,"name"),email:ez(t,"email"),message:ez(t,"message"),attachments:n?[n]:void 0};if(!$(r))return;try{await o({name:r.name,email:r.email,message:r.message,source:"widget",tags:s},{attachments:r.attachments}),i(r)}catch(e){H(e),a(e)}}finally{S(!1)}},[c&&F,i,a])},T&&F?Y(T,{onError:P}):null,Y("fieldset",{class:"form__right","data-sentry-feedback":!0,disabled:C},Y("div",{class:"form__top"},E?Y("div",{class:"form__error-container"},E):null,u?Y("label",{for:"name",class:"form__label"},Y(eq,{label:y,isRequiredLabel:k,isRequired:m}),Y("input",{class:"form__input",defaultValue:n,id:"name",name:"name",placeholder:x,required:m,type:"text"})):Y("input",{"aria-hidden":!0,value:n,name:"name",type:"hidden"}),l?Y("label",{for:"email",class:"form__label"},Y(eq,{label:p,isRequiredLabel:k,isRequired:g}),Y("input",{class:"form__input",defaultValue:t,id:"email",name:"email",placeholder:h,required:g,type:"email"})):Y("input",{"aria-hidden":!0,value:t,name:"email",type:"hidden"}),Y("label",{for:"message",class:"form__label"},Y(eq,{label:v,isRequiredLabel:k,isRequired:!0}),Y("textarea",{autoFocus:!0,class:"form__input form__input--textarea",id:"message",name:"message",placeholder:b,required:!0,rows:5})),T?Y("label",{for:"screenshot",class:"form__label"},Y("button",{class:"btn btn--default",disabled:C,type:"button",onClick:()=>{M(null),L(e=>!e)}},F?d:_),D?Y("div",{class:"form__error-container"},D.message):null):null),Y("div",{class:"btn-group"},Y("button",{class:"btn btn--primary",disabled:C,type:"submit"},w),Y("button",{class:"btn btn--default",disabled:C,type:"button",onClick:r},f))))}function eq({label:e,isRequired:t,isRequiredLabel:n}){return Y("span",{class:"form__label__text"},e,t&&Y("span",{class:"form__label__text--required"},n))}function eI({open:e,onFormSubmitted:t,...n}){let r=n.options,o=eT(()=>({__html:function(){let e=e=>c.document.createElementNS("http://www.w3.org/2000/svg",e),t=L(e("svg"),{width:"16",height:"17",viewBox:"0 0 16 17",fill:"inherit"}),n=L(e("g"),{clipPath:"url(#clip0_57_156)"}),r=L(e("path"),{"fill-rule":"evenodd","clip-rule":"evenodd",d:"M3.55544 15.1518C4.87103 16.0308 6.41775 16.5 8 16.5C10.1217 16.5 12.1566 15.6571 13.6569 14.1569C15.1571 12.6566 16 10.6217 16 8.5C16 6.91775 15.5308 5.37103 14.6518 4.05544C13.7727 2.73985 12.5233 1.71447 11.0615 1.10897C9.59966 0.503466 7.99113 0.34504 6.43928 0.653721C4.88743 0.962403 3.46197 1.72433 2.34315 2.84315C1.22433 3.96197 0.462403 5.38743 0.153721 6.93928C-0.15496 8.49113 0.00346625 10.0997 0.608967 11.5615C1.21447 13.0233 2.23985 14.2727 3.55544 15.1518ZM4.40546 3.1204C5.46945 2.40946 6.72036 2.03 8 2.03C9.71595 2.03 11.3616 2.71166 12.575 3.92502C13.7883 5.13838 14.47 6.78405 14.47 8.5C14.47 9.77965 14.0905 11.0306 13.3796 12.0945C12.6687 13.1585 11.6582 13.9878 10.476 14.4775C9.29373 14.9672 7.99283 15.0953 6.73777 14.8457C5.48271 14.596 4.32987 13.9798 3.42502 13.075C2.52018 12.1701 1.90397 11.0173 1.65432 9.76224C1.40468 8.50718 1.5328 7.20628 2.0225 6.02404C2.5122 4.8418 3.34148 3.83133 4.40546 3.1204Z"}),o=L(e("path"),{d:"M6.68775 12.4297C6.78586 12.4745 6.89218 12.4984 7 12.5C7.11275 12.4955 7.22315 12.4664 7.32337 12.4145C7.4236 12.3627 7.51121 12.2894 7.58 12.2L12 5.63999C12.0848 5.47724 12.1071 5.28902 12.0625 5.11098C12.0178 4.93294 11.9095 4.77744 11.7579 4.67392C11.6064 4.57041 11.4221 4.52608 11.24 4.54931C11.0579 4.57254 10.8907 4.66173 10.77 4.79999L6.88 10.57L5.13 8.56999C5.06508 8.49566 4.98613 8.43488 4.89768 8.39111C4.80922 8.34735 4.713 8.32148 4.61453 8.31498C4.51605 8.30847 4.41727 8.32147 4.32382 8.35322C4.23038 8.38497 4.14413 8.43484 4.07 8.49999C3.92511 8.63217 3.83692 8.81523 3.82387 9.01092C3.81083 9.2066 3.87393 9.39976 4 9.54999L6.43 12.24C6.50187 12.3204 6.58964 12.385 6.68775 12.4297Z"});t.appendChild(n).append(o,r);let i=e("defs"),a=L(e("clipPath"),{id:"clip0_57_156"}),l=L(e("rect"),{width:"16",height:"16",fill:"white",transform:"translate(0 0.5)"});return a.appendChild(l),i.appendChild(a),t.appendChild(i).appendChild(a).appendChild(l),t}().outerHTML}),[]),[i,a]=eH(null),l=eD(()=>{i&&(clearTimeout(i),a(null)),t()},[i]),u=eD(e=>{n.onSubmitSuccess(e),a(setTimeout(()=>{t(),a(null)},5e3))},[t]);return Y(Q,null,i?Y("div",{class:"success__position",onClick:l},Y("div",{class:"success__content"},r.successMessageText,Y("span",{class:"success__icon",dangerouslySetInnerHTML:o}))):Y("dialog",{class:"dialog",onClick:r.onFormClose,open:e},Y("div",{class:"dialog__position"},Y("div",{class:"dialog__content",onClick:e=>{e.stopPropagation()}},Y(eB,{options:r}),Y(eV,{...n,onSubmitSuccess:u})))))}let eW=`
.dialog {
  position: fixed;
  z-index: var(--z-index);
  margin: 0;
  inset: 0;

  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  height: 100vh;
  width: 100vw;

  color: var(--dialog-color, var(--foreground));
  fill: var(--dialog-color, var(--foreground));
  line-height: 1.75em;

  background-color: rgba(0, 0, 0, 0.05);
  border: none;
  inset: 0;
  opacity: 1;
  transition: opacity 0.2s ease-in-out;
}

.dialog__position {
  position: fixed;
  z-index: var(--z-index);
  inset: var(--dialog-inset);
  padding: var(--page-margin);
  display: flex;
  max-height: calc(100vh - (2 * var(--page-margin)));
}
@media (max-width: 600px) {
  .dialog__position {
    inset: var(--page-margin);
    padding: 0;
  }
}

.dialog__position:has(.editor) {
  inset: var(--page-margin);
  padding: 0;
}

.dialog:not([open]) {
  opacity: 0;
  pointer-events: none;
  visibility: hidden;
}
.dialog:not([open]) .dialog__content {
  transform: translate(0, -16px) scale(0.98);
}

.dialog__content {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: var(--dialog-padding, 24px);
  max-width: 100%;
  width: 100%;
  max-height: 100%;
  overflow: auto;

  background: var(--dialog-background, var(--background));
  border-radius: var(--dialog-border-radius, 20px);
  border: var(--dialog-border, var(--border));
  box-shadow: var(--dialog-box-shadow, var(--box-shadow));
  transform: translate(0, 0) scale(1);
  transition: transform 0.2s ease-in-out;
}

`,eO=`
.dialog__header {
  display: flex;
  gap: 4px;
  justify-content: space-between;
  font-weight: var(--dialog-header-weight, 600);
  margin: 0;
}
.dialog__title {
  align-self: center;
  width: var(--form-width, 272px);
}

@media (max-width: 600px) {
  .dialog__title {
    width: auto;
  }
}

.dialog__position:has(.editor) .dialog__title {
  width: auto;
}


.brand-link {
  display: inline-flex;
}
.brand-link:focus-visible {
  outline: var(--outline);
}
`,ej=`
.form {
  display: flex;
  overflow: auto;
  flex-direction: row;
  gap: 16px;
  flex: 1 0;
}

.form fieldset {
  border: none;
  margin: 0;
  padding: 0;
}

.form__right {
  flex: 0 0 auto;
  display: flex;
  overflow: auto;
  flex-direction: column;
  justify-content: space-between;
  gap: 20px;
  width: var(--form-width, 100%);
}

.dialog__position:has(.editor) .form__right {
  width: var(--form-width, 272px);
}

.form__top {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form__error-container {
  color: var(--error-color);
  fill: var(--error-color);
}

.form__label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0px;
}

.form__label__text {
  display: flex;
  gap: 4px;
  align-items: center;
}

.form__label__text--required {
  font-size: 0.85em;
}

.form__input {
  font-family: inherit;
  line-height: inherit;
  background: transparent;
  box-sizing: border-box;
  border: var(--input-border, var(--border));
  border-radius: var(--input-border-radius, 6px);
  color: var(--input-color, inherit);
  fill: var(--input-color, inherit);
  font-size: var(--input-font-size, inherit);
  font-weight: var(--input-font-weight, 500);
  padding: 6px 12px;
}

.form__input::placeholder {
  opacity: 0.65;
  color: var(--input-placeholder-color, inherit);
  filter: var(--interactive-filter);
}

.form__input:focus-visible {
  outline: var(--input-focus-outline, var(--outline));
}

.form__input--textarea {
  font-family: inherit;
  resize: vertical;
}

.error {
  color: var(--error-color);
  fill: var(--error-color);
}
`,eZ=`
.btn-group {
  display: grid;
  gap: 8px;
}

.btn {
  line-height: inherit;
  border: var(--button-border, var(--border));
  border-radius: var(--button-border-radius, 6px);
  cursor: pointer;
  font-family: inherit;
  font-size: var(--button-font-size, inherit);
  font-weight: var(--button-font-weight, 600);
  padding: var(--button-padding, 6px 16px);
}
.btn[disabled] {
  opacity: 0.6;
  pointer-events: none;
}

.btn--primary {
  color: var(--button-primary-color, var(--accent-foreground));
  fill: var(--button-primary-color, var(--accent-foreground));
  background: var(--button-primary-background, var(--accent-background));
  border: var(--button-primary-border, var(--border));
  border-radius: var(--button-primary-border-radius, 6px);
  font-weight: var(--button-primary-font-weight, 500);
}
.btn--primary:hover {
  color: var(--button-primary-hover-color, var(--accent-foreground));
  fill: var(--button-primary-hover-color, var(--accent-foreground));
  background: var(--button-primary-hover-background, var(--accent-background));
  filter: var(--interactive-filter);
}
.btn--primary:focus-visible {
  background: var(--button-primary-hover-background, var(--accent-background));
  filter: var(--interactive-filter);
  outline: var(--button-primary-focus-outline, var(--outline));
}

.btn--default {
  color: var(--button-color, var(--foreground));
  fill: var(--button-color, var(--foreground));
  background: var(--button-background, var(--background));
  border: var(--button-border, var(--border));
  border-radius: var(--button-border-radius, 6px);
  font-weight: var(--button-font-weight, 500);
}
.btn--default:hover {
  color: var(--button-color, var(--foreground));
  fill: var(--button-color, var(--foreground));
  background: var(--button-hover-background, var(--background));
  filter: var(--interactive-filter);
}
.btn--default:focus-visible {
  background: var(--button-hover-background, var(--background));
  filter: var(--interactive-filter);
  outline: var(--button-focus-outline, var(--outline));
}
`,eK=`
.success__position {
  position: fixed;
  inset: var(--dialog-inset);
  padding: var(--page-margin);
  z-index: var(--z-index);
}
.success__content {
  background: var(--success-background, var(--background));
  border: var(--success-border, var(--border));
  border-radius: var(--success-border-radius, 1.7em/50%);
  box-shadow: var(--success-box-shadow, var(--box-shadow));
  font-weight: var(--success-font-weight, 600);
  color: var(--success-color);
  fill: var(--success-color);
  padding: 12px 24px;
  line-height: 1.75em;

  display: grid;
  align-items: center;
  grid-auto-flow: column;
  gap: 6px;
  cursor: default;
}

.success__icon {
  display: flex;
}
`,eY=()=>({name:"FeedbackModal",setupOnce(){},createDialog:({options:e,screenshotIntegration:t,sendFeedback:n,shadow:r})=>{let i=e.useSentryUser,a=function(){let e=(0,o.o5)().getUser(),t=(0,o.rm)().getUser(),n=(0,o.m6)().getUser();return e&&Object.keys(e).length?e:t&&Object.keys(t).length?t:n}(),l=s.createElement("div"),u=function(e){let t=s.createElement("style");return t.textContent=`
:host {
  --dialog-inset: var(--inset);
}

${eW}
${eO}
${ej}
${eZ}
${eK}
`,e&&t.setAttribute("nonce",e),t}(e.styleNonce),c="",_={get el(){return l},appendToDom(){r.contains(u)||r.contains(l)||(r.appendChild(u),r.appendChild(l))},removeFromDom(){l.remove(),u.remove(),s.body.style.overflow=c},open(){f(!0),e.onFormOpen?.(),(0,o.KU)()?.emit("openFeedbackWidget"),c=s.body.style.overflow,s.body.style.overflow="hidden"},close(){f(!1),s.body.style.overflow=c}},d=t?.createInput({h:Y,hooks:eU,dialog:_,options:e}),f=t=>{!function(e,t,n){var r,o,i;R.__&&R.__(e,t),r=t.__k,o=[],i=[],el(t,e=t.__k=Y(Q,null,[e]),r||I,I,void 0!==t.ownerSVGElement,r?null:t.firstChild?A.call(t.childNodes):null,o,r?r.__e:t.firstChild,!1,i),e.__d=void 0,eu(o,e,i)}(Y(eI,{options:e,screenshotInput:d,showName:e.showName||e.isNameRequired,showEmail:e.showEmail||e.isEmailRequired,defaultName:i&&a&&a[i.name]||"",defaultEmail:i&&a&&a[i.email]||"",onFormClose:()=>{f(!1),e.onFormClose?.()},onSubmit:n,onSubmitSuccess:t=>{f(!1),e.onSubmitSuccess?.(t)},onSubmitError:t=>{e.onSubmitError?.(t)},onFormSubmitted:()=>{e.onFormSubmitted?.()},open:t}),l)};return _}});function eG(e,t,n){if(!e)return;let r=e.getContext("2d",t);r&&n(e,r)}function eQ(e,t){eG(e,{alpha:!0},(e,n)=>{n.drawImage(t,0,0,t.width,t.height,0,0,e.width,e.height)})}function eX(e,t,n){eG(e,{alpha:!0},(e,r)=>{n.length&&(r.fillStyle="rgba(0, 0, 0, 0.25)",r.fillRect(0,0,e.width,e.height)),n.forEach(e=>{switch(e.type){case"highlight":r.shadowColor="rgba(0, 0, 0, 0.7)",r.shadowBlur=50,r.fillStyle=t,r.fillRect(e.x-1,e.y-1,e.w+2,e.h+2),r.clearRect(e.x,e.y,e.w,e.h);break;case"hide":r.fillStyle="rgb(0, 0, 0)",r.fillRect(e.x,e.y,e.w,e.h)}})})}let eJ=()=>({name:"FeedbackScreenshot",setupOnce(){},createInput:({h:e,hooks:t,dialog:n,options:r})=>{let o=s.createElement("canvas");return{input:function({h:e,hooks:t,outputBuffer:n,dialog:r,options:o}){let i=function({hooks:e}){return function({onBeforeScreenshot:t,onScreenshot:n,onAfterScreenshot:r,onError:o}){let i=function(){let[t,n]=e.useState(c.devicePixelRatio??1);return e.useEffect(()=>{let e=()=>{n(c.devicePixelRatio)},t=matchMedia(`(resolution: ${c.devicePixelRatio}dppx)`);return t.addEventListener("change",e),()=>{t.removeEventListener("change",e)}},[]),t}();e.useEffect(()=>{(async()=>{t();let e=await _.mediaDevices.getDisplayMedia({video:{width:c.innerWidth*i,height:c.innerHeight*i},audio:!1,monitorTypeSurfaces:"exclude",preferCurrentTab:!0,selfBrowserSurface:"include",surfaceSwitching:"exclude"}),o=s.createElement("video");await new Promise((t,r)=>{o.srcObject=e,o.onloadedmetadata=()=>{n(o,i),e.getTracks().forEach(e=>e.stop()),t()},o.play().catch(r)}),r()})().catch(o)},[])}}({hooks:t}),a=function({h:e}){return function({action:t,setAction:n}){return e("div",{class:"editor__tool-container"},e("div",{class:"editor__tool-bar"},e("button",{type:"button",class:`editor__tool ${"highlight"===t?"editor__tool--active":""}`,onClick:()=>{n("highlight"===t?"":"highlight")}},"Highlight"),e("button",{type:"button",class:`editor__tool ${"hide"===t?"editor__tool--active":""}`,onClick:()=>{n("hide"===t?"":"hide")}},"Hide")))}}({h:e}),l=function({h:e}){return function(){return e("svg",{"data-test-id":"icon-close",viewBox:"0 0 16 16",fill:"#2B2233",height:"25px",width:"25px"},e("circle",{r:"7",cx:"8",cy:"8",fill:"white"}),e("path",{strokeWidth:"1.5",d:"M8,16a8,8,0,1,1,8-8A8,8,0,0,1,8,16ZM8,1.53A6.47,6.47,0,1,0,14.47,8,6.47,6.47,0,0,0,8,1.53Z"}),e("path",{strokeWidth:"1.5",d:"M5.34,11.41a.71.71,0,0,1-.53-.22.74.74,0,0,1,0-1.06l5.32-5.32a.75.75,0,0,1,1.06,1.06L5.87,11.19A.74.74,0,0,1,5.34,11.41Z"}),e("path",{strokeWidth:"1.5",d:"M10.66,11.41a.74.74,0,0,1-.53-.22L4.81,5.87A.75.75,0,0,1,5.87,4.81l5.32,5.32a.74.74,0,0,1,0,1.06A.71.71,0,0,1,10.66,11.41Z"}))}}({h:e}),u={__html:function(e){let t=s.createElement("style"),n="#1A141F",r="#302735";return t.textContent=`
.editor {
  display: flex;
  flex-grow: 1;
  flex-direction: column;
}

.editor__image-container {
  justify-items: center;
  padding: 15px;
  position: relative;
  height: 100%;
  border-radius: var(--menu-border-radius, 6px);

  background-color: ${n};
  background-image: repeating-linear-gradient(
      -145deg,
      transparent,
      transparent 8px,
      ${n} 8px,
      ${n} 11px
    ),
    repeating-linear-gradient(
      -45deg,
      transparent,
      transparent 15px,
      ${r} 15px,
      ${r} 16px
    );
}

.editor__canvas-container {
  width: 100%;
  height: 100%;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.editor__canvas-container > * {
  object-fit: contain;
  position: absolute;
}

.editor__tool-container {
  padding-top: 8px;
  display: flex;
  justify-content: center;
}

.editor__tool-bar {
  display: flex;
  gap: 8px;
}

.editor__tool {
  display: flex;
  padding: 8px 12px;
  justify-content: center;
  align-items: center;
  border: var(--button-border, var(--border));
  border-radius: var(--button-border-radius, 6px);
  background: var(--button-background, var(--background));
  color: var(--button-color, var(--foreground));
}

.editor__tool--active {
  background: var(--button-primary-background, var(--accent-background));
  color: var(--button-primary-color, var(--accent-foreground));
}

.editor__rect {
  position: absolute;
  z-index: 2;
}

.editor__rect button {
  opacity: 0;
  position: absolute;
  top: -12px;
  right: -12px;
  cursor: pointer;
  padding: 0;
  z-index: 3;
  border: none;
  background: none;
}

.editor__rect:hover button {
  opacity: 1;
}
`,e&&t.setAttribute("nonce",e),t}(o.styleNonce).innerText},d=r.el.style,f=({screenshot:r})=>{let[i,_]=t.useState("highlight"),[d,f]=t.useState([]),p=t.useRef(null),h=t.useRef(null),g=t.useRef(null),m=t.useRef(null),[v,b]=t.useState(1),y=t.useMemo(()=>{let e=s.getElementById(o.id);if(!e)return"white";let t=getComputedStyle(e);return t.getPropertyValue("--button-primary-background")||t.getPropertyValue("--accent-background")},[o.id]);t.useLayoutEffect(()=>{let e=()=>{let e=p.current;e&&eG(r.canvas,{alpha:!1},t=>{b(Math.min(e.clientWidth/t.width,e.clientHeight/t.height))})};return e(),c.addEventListener("resize",e),()=>{c.removeEventListener("resize",e)}},[r]);let x=t.useCallback((e,t)=>{eG(e,{alpha:!0},(e,n)=>{n.scale(t,t),e.width=r.canvas.width,e.height=r.canvas.height})},[r]);t.useEffect(()=>{x(h.current,r.dpi),eQ(h.current,r.canvas)},[r]),t.useEffect(()=>{x(g.current,r.dpi),eG(g.current,{alpha:!0},(e,t)=>{t.clearRect(0,0,e.width,e.height)}),eX(g.current,y,d)},[d,y]),t.useEffect(()=>{x(n,r.dpi),eQ(n,r.canvas),eG(s.createElement("canvas"),{alpha:!0},(e,t)=>{t.scale(r.dpi,r.dpi),e.width=r.canvas.width,e.height=r.canvas.height,eX(e,y,d),eQ(n,e)})},[d,r,y]);let w=t.useCallback(e=>t=>{t.preventDefault(),t.stopPropagation(),f(t=>{let n=[...t];return n.splice(e,1),n})},[]),k={width:`${r.canvas.width*v}px`,height:`${r.canvas.height*v}px`},C=e=>{e.stopPropagation()};return e("div",{class:"editor"},e("style",{nonce:o.styleNonce,dangerouslySetInnerHTML:u}),e("div",{class:"editor__image-container"},e("div",{class:"editor__canvas-container",ref:p},e("canvas",{ref:h,id:"background",style:k}),e("canvas",{ref:g,id:"foreground",style:k}),e("div",{ref:m,onMouseDown:e=>{if(!i||!m.current)return;let t=m.current.getBoundingClientRect(),n={type:i,x:e.offsetX/v,y:e.offsetY/v},r=(e,n)=>{let r=(n.clientX-t.x)/v,o=(n.clientY-t.y)/v;return{type:e.type,x:Math.min(e.x,r),y:Math.min(e.y,o),w:Math.abs(r-e.x),h:Math.abs(o-e.y)}},o=e=>{eG(g.current,{alpha:!0},(e,t)=>{t.clearRect(0,0,e.width,e.height)}),eX(g.current,y,[...d,r(n,e)])},a=e=>{let t=r(n,e);t.w*v>=1&&t.h*v>=1&&f(e=>[...e,t]),s.removeEventListener("mousemove",o),s.removeEventListener("mouseup",a)};s.addEventListener("mousemove",o),s.addEventListener("mouseup",a)},style:k},d.map((t,n)=>e("div",{key:n,class:"editor__rect",style:{top:`${t.y*v}px`,left:`${t.x*v}px`,width:`${t.w*v}px`,height:`${t.h*v}px`}},e("button",{"aria-label":"Remove",onClick:w(n),onMouseDown:C,onMouseUp:C,type:"button"},e(l,null))))))),e(a,{action:i,setAction:_}))};return function({onError:r}){let[o,a]=t.useState();return(i({onBeforeScreenshot:t.useCallback(()=>{d.display="none"},[]),onScreenshot:t.useCallback((e,t)=>{eG(s.createElement("canvas"),{alpha:!1},(n,r)=>{r.scale(t,t),n.width=e.videoWidth,n.height=e.videoHeight,r.drawImage(e,0,0,n.width,n.height),a({canvas:n,dpi:t})}),n.width=e.videoWidth,n.height=e.videoHeight},[]),onAfterScreenshot:t.useCallback(()=>{d.display="block"},[]),onError:t.useCallback(e=>{d.display="block",r(e)},[])}),o)?e(f,{screenshot:o}):e("div",null)}}({h:e,hooks:t,outputBuffer:o,dialog:n,options:r}),value:async()=>{let e=await new Promise(e=>{o.toBlob(e,"image/png")});if(e)return{data:new Uint8Array(await e.arrayBuffer()),filename:"screenshot.png",contentType:"application/png"}}}}})}}]);